use core::fmt;
use std::collections::HashMap;

#[cfg(test)]
use symtn_core::RuleIdentity;
use symtn_core::{
    BlockKey, CategoricalScalar, CoreError, FusionTreeKey, MultiplicityFreeRigidSymbols,
    PhysicalFusionBasis, SectorId, SectorLeg,
};

use crate::{
    BoundDynamicFusionMapSpace, BoundDynamicTensorRef, OperationError, RecouplingCoefficientAction,
    TreeTransformScalar,
};

/// Failure from the internal Host physical-basis conversion seam.
#[doc(hidden)]
#[derive(Debug)]
pub enum PhysicalConversionError<E> {
    Operation(OperationError),
    Provider(E),
}

impl<E: fmt::Display> fmt::Display for PhysicalConversionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operation(_) => formatter.write_str("tensor operation failed"),
            Self::Provider(_) => formatter.write_str("fusion provider failed"),
        }
    }
}

impl<E> std::error::Error for PhysicalConversionError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(error) => Some(error),
            Self::Provider(error) => Some(error),
        }
    }
}

impl<E> From<CoreError> for PhysicalConversionError<E> {
    fn from(error: CoreError) -> Self {
        Self::Operation(OperationError::from_core_preserving_context(error))
    }
}

impl<E> From<OperationError> for PhysicalConversionError<E> {
    fn from(error: OperationError) -> Self {
        Self::Operation(error)
    }
}

/// Owned runtime-rank Host data returned by the internal expansion kernel.
#[doc(hidden)]
pub type PhysicalHostBuffer<D> = (Vec<usize>, Vec<D>);

/// Immutable provider-bound plan for repeated physical-basis expansion.
///
/// Provider answers and layout arithmetic are staged once at construction;
/// replay only reads the staged embeddings and the input data.
pub struct PhysicalExpansionPlan<R, C> {
    space: BoundDynamicFusionMapSpace<R>,
    stage: PhysicalStage<C>,
}

impl<R, C> PhysicalExpansionPlan<R, C>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
{
    /// Builds a plan from a validated provider-bound dynamic space.
    pub fn compile(
        space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<Self, PhysicalConversionError<R::Error>> {
        Ok(Self {
            space: space.clone(),
            stage: stage_physical(space)?,
        })
    }

    #[cfg(test)]
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.stage.shape
    }

    #[cfg(test)]
    /// Stable semantic identity of the provider used to compile this plan.
    pub fn provider_identity(&self) -> RuleIdentity {
        self.space.provider().rule_identity()
    }

    /// Expands a tensor bound to the same space/provider allocation.
    pub fn expand_host<D>(
        &self,
        source: BoundDynamicTensorRef<'_, R, D>,
    ) -> Result<PhysicalHostBuffer<D>, PhysicalConversionError<R::Error>>
    where
        D: TreeTransformScalar + RecouplingCoefficientAction<C>,
    {
        if source.space().space() != self.space.space()
            || !std::sync::Arc::ptr_eq(source.space().provider_arc(), self.space.provider_arc())
        {
            return Err(OperationError::StructureMismatch {
                tensor: "physical expansion plan",
            }
            .into());
        }
        expand_physical_stage(&self.stage, source)
    }
}

#[derive(Clone)]
struct LocalTensor<C> {
    left_dim: usize,
    right_dim: usize,
    coupled_dim: usize,
    values: Vec<C>,
}

impl<C> LocalTensor<C> {
    #[inline]
    fn get(&self, left: usize, right: usize, coupled: usize) -> &C {
        &self.values[left + self.left_dim * (right + self.right_dim * coupled)]
    }
}

#[derive(Clone)]
struct DualMap<C> {
    output_dim: usize,
    input_dim: usize,
    values: Vec<C>,
}

impl<C> DualMap<C> {
    #[inline]
    fn get(&self, output: usize, input: usize) -> &C {
        &self.values[output + self.output_dim * input]
    }
}

#[derive(Clone)]
struct TreeEmbedding<C> {
    physical_dims: Vec<usize>,
    physical_len: usize,
    coupled_dim: usize,
    values: Vec<C>,
}

impl<C> TreeEmbedding<C> {
    #[inline]
    fn get(&self, physical: usize, coupled: usize) -> &C {
        &self.values[physical + self.physical_len * coupled]
    }
}

#[derive(Clone, Copy)]
struct SectorSlice {
    offset: usize,
    degeneracy: usize,
    carrier_dim: usize,
}

struct LegLayout {
    dimension: usize,
    sectors: HashMap<SectorId, SectorSlice>,
}

struct StagedBlock<C> {
    codomain_embedding: usize,
    domain_embedding: usize,
    normalization: C,
    slices: Vec<SectorSlice>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

struct PhysicalStage<C> {
    shape: Vec<usize>,
    dense_strides: Vec<usize>,
    dense_len: usize,
    required_len: usize,
    embeddings: Vec<TreeEmbedding<C>>,
    blocks: Vec<StagedBlock<C>>,
}

struct BasisStage<'a, R, C> {
    rule: &'a R,
    carrier_dims: HashMap<SectorId, usize>,
    locals: HashMap<(SectorId, SectorId, SectorId, usize), LocalTensor<C>>,
    duals: HashMap<SectorId, DualMap<C>>,
}

impl<'a, R, C> BasisStage<'a, R, C>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
{
    fn new(rule: &'a R) -> Self {
        Self {
            rule,
            carrier_dims: HashMap::new(),
            locals: HashMap::new(),
            duals: HashMap::new(),
        }
    }

    fn carrier_dim(
        &mut self,
        sector: SectorId,
    ) -> Result<usize, PhysicalConversionError<R::Error>> {
        if let Some(&dimension) = self.carrier_dims.get(&sector) {
            return Ok(dimension);
        }
        let dimension = self
            .rule
            .try_carrier_dimension(sector)
            .map_err(PhysicalConversionError::Provider)?;
        self.carrier_dims.insert(sector, dimension);
        Ok(dimension)
    }

    fn local(
        &mut self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        multiplicity: usize,
    ) -> Result<&LocalTensor<C>, PhysicalConversionError<R::Error>> {
        let key = (left, right, coupled, multiplicity);
        if !self.locals.contains_key(&key) {
            let left_dim = self.carrier_dim(left)?;
            let right_dim = self.carrier_dim(right)?;
            let coupled_dim = self.carrier_dim(coupled)?;
            let len = checked_product([left_dim, right_dim, coupled_dim])?;
            let mut values = Vec::with_capacity(len);
            for coupled_basis in 0..coupled_dim {
                for right_basis in 0..right_dim {
                    for left_basis in 0..left_dim {
                        values.push(
                            self.rule
                                .try_fusion_tensor_element(
                                    left,
                                    right,
                                    coupled,
                                    left_basis,
                                    right_basis,
                                    coupled_basis,
                                    multiplicity,
                                )
                                .map_err(PhysicalConversionError::Provider)?,
                        );
                    }
                }
            }
            self.locals.insert(
                key,
                LocalTensor {
                    left_dim,
                    right_dim,
                    coupled_dim,
                    values,
                },
            );
        }
        Ok(self.locals.get(&key).expect("local tensor was just staged"))
    }

    fn dual_map(
        &mut self,
        sector: SectorId,
    ) -> Result<&DualMap<C>, PhysicalConversionError<R::Error>> {
        if !self.duals.contains_key(&sector) {
            let dual = self.rule.dual(sector);
            let vacuum = self.rule.vacuum();
            let local = self.local(dual, sector, vacuum, 0)?.clone();
            if local.coupled_dim != 1 {
                return Err(OperationError::InvalidArgument {
                    message: "physical tensor unit must have a one-dimensional carrier basis",
                }
                .into());
            }
            let output_dim = local.left_dim;
            let input_dim = local.right_dim;
            let sqrt_dim = self.rule.sqrt_dim_scalar(sector);
            let mut values = Vec::with_capacity(checked_product([output_dim, input_dim])?);
            for input in 0..input_dim {
                for output in 0..output_dim {
                    values.push((sqrt_dim.clone() * local.get(output, input, 0).clone()).conj());
                }
            }
            self.duals.insert(
                sector,
                DualMap {
                    output_dim,
                    input_dim,
                    values,
                },
            );
        }
        Ok(self.duals.get(&sector).expect("dual map was just staged"))
    }

    fn tree(
        &mut self,
        tree: &FusionTreeKey,
    ) -> Result<TreeEmbedding<C>, PhysicalConversionError<R::Error>> {
        let rank = tree.uncoupled().len();
        if tree.is_dual().len() != rank
            || tree.innerlines().len() != rank.saturating_sub(2)
            || tree.vertices().len() != rank.saturating_sub(1)
        {
            return Err(OperationError::InvalidArgument {
                message: "malformed fusion-tree shape at physical conversion boundary",
            }
            .into());
        }
        match rank {
            0 => self.rank_zero_tree(tree),
            1 => self.rank_one_tree(tree),
            _ => self.multi_leg_tree(tree),
        }
    }

    fn rank_zero_tree(
        &mut self,
        tree: &FusionTreeKey,
    ) -> Result<TreeEmbedding<C>, PhysicalConversionError<R::Error>> {
        let vacuum = self.rule.vacuum();
        let local = self.local(vacuum, vacuum, tree.coupled(), 0)?;
        if local.left_dim != 1 || local.right_dim != 1 {
            return Err(OperationError::InvalidArgument {
                message: "physical tensor unit must have a one-dimensional carrier basis",
            }
            .into());
        }
        Ok(TreeEmbedding {
            physical_dims: Vec::new(),
            physical_len: 1,
            coupled_dim: local.coupled_dim,
            values: (0..local.coupled_dim)
                .map(|coupled| local.get(0, 0, coupled).clone())
                .collect(),
        })
    }

    fn rank_one_tree(
        &mut self,
        tree: &FusionTreeKey,
    ) -> Result<TreeEmbedding<C>, PhysicalConversionError<R::Error>> {
        let sector = tree.uncoupled()[0];
        if tree.is_dual()[0] {
            let dual = self.dual_map(sector)?.clone();
            if dual.input_dim != self.carrier_dim(tree.coupled())? {
                return Err(OperationError::InvalidArgument {
                    message: "dual carrier map does not match the coupled carrier dimension",
                }
                .into());
            }
            return Ok(TreeEmbedding {
                physical_dims: vec![dual.output_dim],
                physical_len: dual.output_dim,
                coupled_dim: dual.input_dim,
                values: dual.values.clone(),
            });
        }

        let vacuum = self.rule.vacuum();
        let local = self.local(sector, vacuum, tree.coupled(), 0)?;
        if local.right_dim != 1 {
            return Err(OperationError::InvalidArgument {
                message: "physical tensor unit must have a one-dimensional carrier basis",
            }
            .into());
        }
        Ok(TreeEmbedding {
            physical_dims: vec![local.left_dim],
            physical_len: local.left_dim,
            coupled_dim: local.coupled_dim,
            values: (0..local.coupled_dim)
                .flat_map(|coupled| {
                    (0..local.left_dim).map(move |left| local.get(left, 0, coupled).clone())
                })
                .collect(),
        })
    }

    fn multi_leg_tree(
        &mut self,
        tree: &FusionTreeKey,
    ) -> Result<TreeEmbedding<C>, PhysicalConversionError<R::Error>> {
        let rank = tree.uncoupled().len();
        let first_coupled = if rank == 2 {
            tree.coupled()
        } else {
            tree.innerlines()[0]
        };
        let first = self
            .local(
                tree.uncoupled()[0],
                tree.uncoupled()[1],
                first_coupled,
                tree.vertices()[0].get() - 1,
            )?
            .clone();
        let mut embedding = self.apply_external_duality(
            first,
            tree.uncoupled()[0],
            tree.is_dual()[0],
            tree.uncoupled()[1],
            tree.is_dual()[1],
        )?;

        for axis in 2..rank {
            let next_coupled = if axis + 1 == rank {
                tree.coupled()
            } else {
                tree.innerlines()[axis - 1]
            };
            let local = self
                .local(
                    if axis == 2 {
                        tree.innerlines()[0]
                    } else {
                        tree.innerlines()[axis - 2]
                    },
                    tree.uncoupled()[axis],
                    next_coupled,
                    tree.vertices()[axis - 1].get() - 1,
                )?
                .clone();
            embedding = self.append_leg(
                embedding,
                local,
                tree.uncoupled()[axis],
                tree.is_dual()[axis],
            )?;
        }
        Ok(embedding)
    }

    fn apply_external_duality(
        &mut self,
        local: LocalTensor<C>,
        left_sector: SectorId,
        left_dual: bool,
        right_sector: SectorId,
        right_dual: bool,
    ) -> Result<TreeEmbedding<C>, PhysicalConversionError<R::Error>> {
        let left_map = left_dual
            .then(|| self.dual_map(left_sector).cloned())
            .transpose()?;
        let right_map = right_dual
            .then(|| self.dual_map(right_sector).cloned())
            .transpose()?;
        let left_dim = left_map
            .as_ref()
            .map_or(local.left_dim, |map| map.output_dim);
        let right_dim = right_map
            .as_ref()
            .map_or(local.right_dim, |map| map.output_dim);
        if left_map
            .as_ref()
            .is_some_and(|map| map.input_dim != local.left_dim)
            || right_map
                .as_ref()
                .is_some_and(|map| map.input_dim != local.right_dim)
        {
            return Err(OperationError::InvalidArgument {
                message: "dual carrier map does not match a local fusion tensor",
            }
            .into());
        }
        let physical_len = checked_product([left_dim, right_dim])?;
        let mut values = Vec::with_capacity(checked_product([physical_len, local.coupled_dim])?);
        for coupled in 0..local.coupled_dim {
            for right_out in 0..right_dim {
                for left_out in 0..left_dim {
                    let mut value = C::zero();
                    for right_in in 0..local.right_dim {
                        let right_factor = right_map.as_ref().map_or_else(
                            || (right_out == right_in).then(C::one).unwrap_or_else(C::zero),
                            |map| map.get(right_out, right_in).clone(),
                        );
                        for left_in in 0..local.left_dim {
                            let left_factor = left_map.as_ref().map_or_else(
                                || (left_out == left_in).then(C::one).unwrap_or_else(C::zero),
                                |map| map.get(left_out, left_in).clone(),
                            );
                            value = value
                                + left_factor.clone()
                                    * right_factor.clone()
                                    * local.get(left_in, right_in, coupled).clone();
                        }
                    }
                    values.push(value);
                }
            }
        }
        Ok(TreeEmbedding {
            physical_dims: vec![left_dim, right_dim],
            physical_len,
            coupled_dim: local.coupled_dim,
            values,
        })
    }

    fn append_leg(
        &mut self,
        embedding: TreeEmbedding<C>,
        local: LocalTensor<C>,
        sector: SectorId,
        is_dual: bool,
    ) -> Result<TreeEmbedding<C>, PhysicalConversionError<R::Error>> {
        if embedding.coupled_dim != local.left_dim {
            return Err(OperationError::InvalidArgument {
                message: "recursive physical fusion tensor has mismatched inner carrier dimensions",
            }
            .into());
        }
        let map = is_dual
            .then(|| self.dual_map(sector).cloned())
            .transpose()?;
        let external_dim = map.as_ref().map_or(local.right_dim, |map| map.output_dim);
        if map
            .as_ref()
            .is_some_and(|map| map.input_dim != local.right_dim)
        {
            return Err(OperationError::InvalidArgument {
                message: "dual carrier map does not match a local fusion tensor",
            }
            .into());
        }
        let physical_len = embedding
            .physical_len
            .checked_mul(external_dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        let mut values = Vec::with_capacity(checked_product([physical_len, local.coupled_dim])?);
        for coupled in 0..local.coupled_dim {
            for external_out in 0..external_dim {
                for old_physical in 0..embedding.physical_len {
                    let mut value = C::zero();
                    for external_in in 0..local.right_dim {
                        let external_factor = map.as_ref().map_or_else(
                            || {
                                (external_out == external_in)
                                    .then(C::one)
                                    .unwrap_or_else(C::zero)
                            },
                            |map| map.get(external_out, external_in).clone(),
                        );
                        for inner in 0..embedding.coupled_dim {
                            value = value
                                + embedding.get(old_physical, inner).clone()
                                    * external_factor.clone()
                                    * local.get(inner, external_in, coupled).clone();
                        }
                    }
                    values.push(value);
                }
            }
        }
        let mut physical_dims = embedding.physical_dims;
        physical_dims.push(external_dim);
        Ok(TreeEmbedding {
            physical_dims,
            physical_len,
            coupled_dim: local.coupled_dim,
            values,
        })
    }
}

fn checked_product<I, E>(factors: I) -> Result<usize, PhysicalConversionError<E>>
where
    I: IntoIterator<Item = usize>,
{
    factors.into_iter().try_fold(1usize, |product, factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| OperationError::ElementCountOverflow.into())
    })
}

fn leg_layout<R, C>(
    basis: &mut BasisStage<'_, R, C>,
    leg: &SectorLeg,
) -> Result<LegLayout, PhysicalConversionError<R::Error>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
{
    // TensorKit `axes(V::GradedSpace, c)` walks `sectors(V)`, which for
    // `V'` yields `dual(c)` for `c` in `V`'s order; `SectorId` order would
    // make bending a leg reorder its dense axis.
    let rule = basis.rule;
    let mut ordered = leg.iter().collect::<Vec<_>>();
    ordered.sort_by_cached_key(|&(sector, _)| {
        rule.sector_order_key(if leg.is_dual() {
            rule.dual(sector)
        } else {
            sector
        })
    });
    let mut dimension = 0usize;
    let mut sectors = HashMap::with_capacity(ordered.len());
    for (sector, degeneracy) in ordered {
        let carrier_dim = basis.carrier_dim(sector)?;
        let width = degeneracy
            .checked_mul(carrier_dim)
            .ok_or(OperationError::ElementCountOverflow)?;
        sectors.insert(
            sector,
            SectorSlice {
                offset: dimension,
                degeneracy,
                carrier_dim,
            },
        );
        dimension = dimension
            .checked_add(width)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(LegLayout { dimension, sectors })
}

fn stage_physical<R, C>(
    space: &BoundDynamicFusionMapSpace<R>,
) -> Result<PhysicalStage<C>, PhysicalConversionError<R::Error>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
{
    let raw = space.space();
    let rule = space.provider();
    let mut basis = BasisStage::new(rule);
    let leg_layouts = raw
        .homspace()
        .codomain()
        .legs()
        .iter()
        .chain(raw.homspace().domain().legs())
        .map(|leg| leg_layout(&mut basis, leg))
        .collect::<Result<Vec<_>, _>>()?;
    let shape = leg_layouts
        .iter()
        .map(|leg| leg.dimension)
        .collect::<Vec<_>>();
    let dense_len = checked_product(shape.iter().copied())?;
    let mut dense_strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &dimension in &shape {
        dense_strides.push(stride);
        stride = stride
            .checked_mul(dimension)
            .ok_or(OperationError::ElementCountOverflow)?;
    }

    let required_len = raw.required_len()?;
    let structure = raw.structure();
    let mut embedding_indices = HashMap::<FusionTreeKey, usize>::new();
    let mut embeddings = Vec::new();
    let mut blocks = Vec::with_capacity(structure.block_count());
    for block_index in 0..structure.block_count() {
        let block = structure.block(block_index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "physical conversion",
                index: block_index,
            }
            .into());
        };
        let codomain_embedding = ensure_embedding(
            &mut basis,
            &mut embedding_indices,
            &mut embeddings,
            key.codomain_tree(),
        )?;
        let domain_embedding = ensure_embedding(
            &mut basis,
            &mut embedding_indices,
            &mut embeddings,
            key.domain_tree(),
        )?;
        if embeddings[codomain_embedding].coupled_dim != embeddings[domain_embedding].coupled_dim {
            return Err(OperationError::InvalidArgument {
                message: "fusion-tree pair has mismatched coupled carrier dimensions",
            }
            .into());
        }
        let sectors = key
            .codomain_tree()
            .uncoupled()
            .iter()
            .chain(key.domain_tree().uncoupled());
        let slices = leg_layouts
            .iter()
            .zip(sectors)
            .enumerate()
            .map(|(axis, (leg, sector))| {
                leg.sectors
                    .get(sector)
                    .copied()
                    .ok_or_else(|| OperationError::StructureMismatch {
                        tensor: if axis < raw.nout() {
                            "physical codomain leg"
                        } else {
                            "physical domain leg"
                        },
                    })
                    .map_err(Into::into)
            })
            .collect::<Result<Vec<_>, PhysicalConversionError<R::Error>>>()?;
        if block.shape().len() != slices.len()
            || block
                .shape()
                .iter()
                .zip(&slices)
                .any(|(&actual, slice)| actual != slice.degeneracy)
        {
            return Err(OperationError::StructureMismatch {
                tensor: "physical conversion reduced block",
            }
            .into());
        }
        if embeddings[codomain_embedding].physical_dims
            != slices[..raw.nout()]
                .iter()
                .map(|slice| slice.carrier_dim)
                .collect::<Vec<_>>()
            || embeddings[domain_embedding].physical_dims
                != slices[raw.nout()..]
                    .iter()
                    .map(|slice| slice.carrier_dim)
                    .collect::<Vec<_>>()
        {
            return Err(OperationError::StructureMismatch {
                tensor: "physical carrier dimensions",
            }
            .into());
        }
        if block.storage_end_exclusive()? > required_len {
            return Err(OperationError::StructureMismatch {
                tensor: "physical conversion reduced storage",
            }
            .into());
        }
        blocks.push(StagedBlock {
            codomain_embedding,
            domain_embedding,
            normalization: rule.inv_dim_scalar(key.coupled()),
            slices,
            shape: block.shape().to_vec(),
            strides: block.strides().to_vec(),
            offset: block.offset(),
        });
    }
    Ok(PhysicalStage {
        shape,
        dense_strides,
        dense_len,
        required_len,
        embeddings,
        blocks,
    })
}

fn ensure_embedding<R, C>(
    basis: &mut BasisStage<'_, R, C>,
    indices: &mut HashMap<FusionTreeKey, usize>,
    embeddings: &mut Vec<TreeEmbedding<C>>,
    tree: &FusionTreeKey,
) -> Result<usize, PhysicalConversionError<R::Error>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
{
    if let Some(&index) = indices.get(tree) {
        return Ok(index);
    }
    let embedding = basis.tree(tree)?;
    let index = embeddings.len();
    embeddings.push(embedding);
    indices.insert(tree.clone(), index);
    Ok(index)
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    if shape.is_empty() {
        visit(&[]);
        return;
    }
    if shape.contains(&0) {
        return;
    }
    let rank = shape.len();
    let mut index = vec![0usize; rank];
    loop {
        visit(&index);
        let mut axis = 0;
        loop {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
            axis += 1;
            if axis == rank {
                return;
            }
        }
    }
}

fn reduced_position<C>(block: &StagedBlock<C>, index: &[usize]) -> usize {
    // `stage_physical` checked this block's exact storage end through the
    // validated BlockStructure before execution, so these products and sums
    // are bounded by `required_len`.
    block.offset
        + index
            .iter()
            .zip(&block.strides)
            .map(|(&index, &stride)| index * stride)
            .sum::<usize>()
}

fn dense_position(
    stage: &PhysicalStage<impl Clone>,
    block: &StagedBlock<impl Clone>,
    degeneracy: &[usize],
    carrier: &[usize],
) -> usize {
    // The staged physical shape product bounds every coordinate/stride term by
    // `dense_len`; execution cannot overflow after that checked preflight.
    block
        .slices
        .iter()
        .zip(degeneracy)
        .zip(carrier)
        .zip(&stage.dense_strides)
        .map(|(((slice, &degeneracy), &carrier), &stride)| {
            (slice.offset + degeneracy * slice.carrier_dim + carrier) * stride
        })
        .sum()
}

fn embedding_physical_index(dims: &[usize], carrier: &[usize]) -> usize {
    carrier
        .iter()
        .zip(dims)
        .fold((0usize, 1usize), |(index, stride), (&value, &dimension)| {
            (index + value * stride, stride * dimension)
        })
        .0
}

fn pair_coefficient<C: CategoricalScalar>(
    codomain: &TreeEmbedding<C>,
    domain: &TreeEmbedding<C>,
    carrier: &[usize],
    nout: usize,
) -> C {
    let codomain_physical = embedding_physical_index(&codomain.physical_dims, &carrier[..nout]);
    let domain_physical = embedding_physical_index(&domain.physical_dims, &carrier[nout..]);
    (0..codomain.coupled_dim).fold(C::zero(), |value, coupled| {
        value
            + codomain.get(codomain_physical, coupled).clone()
                * domain.get(domain_physical, coupled).conj()
    })
}

/// Expands one complete reduced dynamic tensor into owned column-major Host data.
#[doc(hidden)]
pub fn expand_physical_host<R, D, C>(
    source: BoundDynamicTensorRef<'_, R, D>,
) -> Result<PhysicalHostBuffer<D>, PhysicalConversionError<R::Error>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
    D: TreeTransformScalar + RecouplingCoefficientAction<C>,
{
    let plan = PhysicalExpansionPlan::<R, C>::compile(source.space())?;
    plan.expand_host(source)
}

fn expand_physical_stage<R, D, C>(
    stage: &PhysicalStage<C>,
    source: BoundDynamicTensorRef<'_, R, D>,
) -> Result<PhysicalHostBuffer<D>, PhysicalConversionError<R::Error>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
    D: TreeTransformScalar + RecouplingCoefficientAction<C>,
{
    if source.data().len() != stage.required_len {
        return Err(OperationError::ElementCountMismatch {
            expected: stage.required_len,
            actual: source.data().len(),
        }
        .into());
    }
    let mut output = vec![D::zero(); stage.dense_len];
    for block in &stage.blocks {
        let codomain = &stage.embeddings[block.codomain_embedding];
        let domain = &stage.embeddings[block.domain_embedding];
        let carrier_shape = block
            .slices
            .iter()
            .map(|slice| slice.carrier_dim)
            .collect::<Vec<_>>();
        for_each_index(&block.shape, |degeneracy| {
            let source_value = source.data()[reduced_position(block, degeneracy)];
            for_each_index(&carrier_shape, |carrier| {
                let coefficient =
                    pair_coefficient(codomain, domain, carrier, source.space().space().nout());
                let position = dense_position(stage, block, degeneracy, carrier);
                output[position] =
                    output[position] + source_value.scale_by_coefficient(coefficient);
            });
        });
    }
    Ok((stage.shape.clone(), output))
}

/// Projects owned/borrowed column-major Host data into one exact dynamic layout.
#[doc(hidden)]
pub fn project_physical_host<R, D, C>(
    destination: &BoundDynamicFusionMapSpace<R>,
    shape: &[usize],
    data: &[D],
) -> Result<Vec<D>, PhysicalConversionError<R::Error>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = C> + PhysicalFusionBasis<Scalar = C>,
    C: CategoricalScalar,
    D: TreeTransformScalar + RecouplingCoefficientAction<C>,
{
    // Stage every provider answer and checked offset before allocating reduced output.
    let stage = stage_physical(destination)?;
    if shape != stage.shape {
        return Err(OperationError::ShapeMismatch {
            dst: stage.shape,
            src: shape.to_vec(),
        }
        .into());
    }
    if data.len() != stage.dense_len {
        return Err(OperationError::ElementCountMismatch {
            expected: stage.dense_len,
            actual: data.len(),
        }
        .into());
    }
    let mut output = vec![D::zero(); stage.required_len];
    for block in &stage.blocks {
        let codomain = &stage.embeddings[block.codomain_embedding];
        let domain = &stage.embeddings[block.domain_embedding];
        let carrier_shape = block
            .slices
            .iter()
            .map(|slice| slice.carrier_dim)
            .collect::<Vec<_>>();
        for_each_index(&block.shape, |degeneracy| {
            let mut value = D::zero();
            for_each_index(&carrier_shape, |carrier| {
                let coefficient =
                    pair_coefficient(codomain, domain, carrier, destination.space().nout()).conj();
                value = value
                    + data[dense_position(&stage, block, degeneracy, carrier)]
                        .scale_by_coefficient(coefficient);
            });
            output[reduced_position(block, degeneracy)] =
                value.scale_by_coefficient(block.normalization.clone());
        });
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use num_complex::Complex64;
    use symtn_core::{
        BlockSpec, BlockStructure, BraidingStyleKind, FusionProductSpace, FusionRule,
        FusionStyleKind, FusionTreeHomSpace, MultiplicityFreeFusionRule,
        MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, PhysicalFusionBasis,
        RuleIdentity, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, SectorVec, U1FusionRule,
        U1Irrep,
    };

    use super::*;
    use crate::DynamicFusionMapSpace;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct ProbePhysicalError;

    impl fmt::Display for ProbePhysicalError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("deliberate physical-basis failure")
        }
    }

    impl std::error::Error for ProbePhysicalError {}

    struct FailingPhysicalU1 {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    impl symtn_core::testing::Provider for FailingPhysicalU1 {}

    impl FusionRule for FailingPhysicalU1 {
        fn rule_identity(&self) -> RuleIdentity {
            U1FusionRule.rule_identity()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            U1FusionRule.fusion_style()
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            U1FusionRule.braiding_style()
        }

        fn vacuum(&self) -> SectorId {
            U1FusionRule.vacuum()
        }

        fn dual(&self, sector: SectorId) -> SectorId {
            U1FusionRule.dual(sector)
        }

        fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
            U1FusionRule.fusion_channels(left, right)
        }

        fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
            U1FusionRule.nsymbol(left, right, coupled)
        }
    }

    impl MultiplicityFreeFusionRule for FailingPhysicalU1 {}

    impl MultiplicityFreeFusionSymbols for FailingPhysicalU1 {
        type Scalar = f64;

        fn has_trivial_associator_gauge(&self) -> bool {
            true
        }

        fn f_symbol_scalar(
            &self,
            left: SectorId,
            middle: SectorId,
            right: SectorId,
            coupled: SectorId,
            left_coupled: SectorId,
            right_coupled: SectorId,
        ) -> f64 {
            U1FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
        }

        fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> f64 {
            U1FusionRule.r_symbol_scalar(left, right, coupled)
        }
    }

    impl MultiplicityFreeRigidSymbols for FailingPhysicalU1 {
        fn dim_scalar(&self, sector: SectorId) -> f64 {
            U1FusionRule.dim_scalar(sector)
        }

        fn inv_dim_scalar(&self, sector: SectorId) -> f64 {
            U1FusionRule.inv_dim_scalar(sector)
        }

        fn sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
            U1FusionRule.sqrt_dim_scalar(sector)
        }

        fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
            U1FusionRule.inv_sqrt_dim_scalar(sector)
        }

        fn twist_scalar(&self, sector: SectorId) -> f64 {
            U1FusionRule.twist_scalar(sector)
        }

        fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> f64 {
            U1FusionRule.frobenius_schur_phase_scalar(sector)
        }
    }

    impl PhysicalFusionBasis for FailingPhysicalU1 {
        type Scalar = f64;
        type Error = ProbePhysicalError;

        fn try_carrier_dimension(&self, _sector: SectorId) -> Result<usize, Self::Error> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(1)
        }

        fn try_fusion_tensor_element(
            &self,
            _left: SectorId,
            _right: SectorId,
            _coupled: SectorId,
            _left_basis: usize,
            _right_basis: usize,
            _coupled_basis: usize,
            _multiplicity: usize,
        ) -> Result<f64, Self::Error> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.fail {
                Err(ProbePhysicalError)
            } else {
                Ok(1.0)
            }
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 2.0e-12,
                "{actual} != {expected}"
            );
        }
    }

    fn su2_multitree_space() -> BoundDynamicFusionMapSpace<SU2FusionRule> {
        let half = SU2Irrep::from_twice_spin(1);
        let leg = || SectorLeg::new([(half, 1)], false);
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(SU2FusionRule),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg(), leg(), leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    }

    #[test]
    fn su2_multitree_real_and_complex_reduced_roundtrip() {
        let space = su2_multitree_space();
        assert_eq!(space.space().structure().block_count(), 2);
        let reduced = vec![1.25, -0.75];
        let source = BoundDynamicTensorRef::try_new(&space, &reduced).unwrap();
        let (shape, physical) = expand_physical_host(source).unwrap();
        assert_eq!(shape, [2, 2, 2, 2]);
        assert_close(
            &project_physical_host(&space, &shape, &physical).unwrap(),
            &reduced,
        );

        let reduced = vec![Complex64::new(1.25, -0.5), Complex64::new(-0.75, 0.25)];
        let source = BoundDynamicTensorRef::try_new(&space, &reduced).unwrap();
        let (shape, physical) = expand_physical_host(source).unwrap();
        let projected = project_physical_host(&space, &shape, &physical).unwrap();
        for (&actual, &expected) in projected.iter().zip(&reduced) {
            assert!((actual - expected).norm() < 2.0e-12);
        }
    }

    #[test]
    fn physical_expansion_plan_reuses_staged_provider_data() {
        let space = su2_multitree_space();
        let plan = PhysicalExpansionPlan::<SU2FusionRule, f64>::compile(&space).unwrap();
        assert_eq!(plan.shape(), &[2, 2, 2, 2]);
        assert_eq!(plan.provider_identity(), space.provider().rule_identity());
        let first = plan
            .expand_host(BoundDynamicTensorRef::try_new(&space, &[1.25, -0.75]).unwrap())
            .unwrap();
        let second = plan
            .expand_host(BoundDynamicTensorRef::try_new(&space, &[0.5, 0.25]).unwrap())
            .unwrap();
        assert_eq!(first.0, second.0);
        assert_ne!(first.1, second.1);
    }

    fn counting_u1_space(
        provider: Arc<FailingPhysicalU1>,
    ) -> BoundDynamicFusionMapSpace<FailingPhysicalU1> {
        let leg = SectorLeg::new([(U1Irrep::new(0), 1)], false);
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            provider,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone()]),
                FusionProductSpace::new([leg]),
            ),
        )
        .unwrap()
    }

    #[test]
    fn physical_expansion_plan_is_provider_staged_and_reusable_for_complex_data() {
        let calls = Arc::new(AtomicUsize::new(0));
        let space = counting_u1_space(Arc::new(FailingPhysicalU1 {
            calls: Arc::clone(&calls),
            fail: false,
        }));
        let plan = PhysicalExpansionPlan::<FailingPhysicalU1, f64>::compile(&space).unwrap();
        let staged_calls = calls.load(Ordering::Relaxed);
        assert!(
            staged_calls > 0,
            "plan compilation must stage provider answers"
        );
        plan.expand_host(BoundDynamicTensorRef::try_new(&space, &[1.0]).unwrap())
            .unwrap();
        plan.expand_host(
            BoundDynamicTensorRef::try_new(&space, &[Complex64::new(2.0, -0.5)]).unwrap(),
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), staged_calls);
    }

    #[test]
    fn physical_expansion_plan_rejects_equivalent_but_distinct_provider_arc() {
        let first = counting_u1_space(Arc::new(FailingPhysicalU1 {
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }));
        let second = counting_u1_space(Arc::new(FailingPhysicalU1 {
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }));
        let plan = PhysicalExpansionPlan::<FailingPhysicalU1, f64>::compile(&first).unwrap();
        assert!(matches!(
            plan.expand_host(BoundDynamicTensorRef::try_new(&second, &[1.0]).unwrap()),
            Err(PhysicalConversionError::Operation(
                OperationError::StructureMismatch { .. }
            ))
        ));
    }

    #[test]
    fn physical_expansion_plan_survives_provenance_failure() {
        let first = counting_u1_space(Arc::new(FailingPhysicalU1 {
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }));
        let second = counting_u1_space(Arc::new(FailingPhysicalU1 {
            calls: Arc::new(AtomicUsize::new(0)),
            fail: false,
        }));
        let plan = PhysicalExpansionPlan::<FailingPhysicalU1, f64>::compile(&first).unwrap();
        assert!(plan
            .expand_host(BoundDynamicTensorRef::try_new(&second, &[1.0]).unwrap())
            .is_err());
        assert!(plan
            .expand_host(BoundDynamicTensorRef::try_new(&first, &[1.0]).unwrap())
            .is_ok());
    }

    #[test]
    fn su2_recursive_embedding_matches_tensorkit_descending_basis_oracle() {
        let space = su2_multitree_space();
        let (_, singlet_path) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&space, &[1.0, 0.0]).unwrap())
                .unwrap();
        let (_, triplet_path) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&space, &[0.0, 1.0]).unwrap())
                .unwrap();

        // TensorKitSectors' executable basis is m=+1/2,-1/2. These are the
        // two left-associated ((1/2 x 1/2) -> 0 or 1) x 1/2 -> 1/2 columns,
        // paired with the one-leg domain identity. The asymmetric signs pin
        // recursion, coupled-index contraction, and carrier ordering rather
        // than merely checking expansion/projection self-consistency.
        let inv_sqrt_2 = 1.0 / 2.0_f64.sqrt();
        let inv_sqrt_6 = 1.0 / 6.0_f64.sqrt();
        assert!((singlet_path[1] + inv_sqrt_2).abs() < 2.0e-12);
        assert!((singlet_path[2] - inv_sqrt_2).abs() < 2.0e-12);
        assert!((singlet_path[13] + inv_sqrt_2).abs() < 2.0e-12);
        assert!((singlet_path[14] - inv_sqrt_2).abs() < 2.0e-12);
        assert!((triplet_path[1] + inv_sqrt_6).abs() < 2.0e-12);
        assert!((triplet_path[2] + inv_sqrt_6).abs() < 2.0e-12);
        assert!((triplet_path[4] - 2.0 * inv_sqrt_6).abs() < 2.0e-12);
        assert!((triplet_path[11] + 2.0 * inv_sqrt_6).abs() < 2.0e-12);
    }

    #[test]
    fn su2_dual_map_is_tensorkit_antisymmetric_z_dagger() {
        let half = SU2Irrep::from_twice_spin(1);
        let space = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(SU2FusionRule),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(half, 1)], true)]),
                FusionProductSpace::new([SectorLeg::new([(half, 1)], false)]),
            ),
        )
        .unwrap();
        let (_, physical) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&space, &[1.0]).unwrap()).unwrap();

        // TensorKit rank-one dual convention:
        // Z_{1/2}^† = conj(sqrt(2) C[1/2,1/2,0]). In the executable
        // descending (+1/2,-1/2) basis this is [[0,1],[-1,0]]. The flat
        // physical tensor is column-major with codomain carrier first.
        assert_close(&physical, &[0.0, -1.0, 1.0, 0.0]);
        assert_close(
            &project_physical_host(&space, &[2, 2], &physical).unwrap(),
            &[1.0],
        );
    }

    #[test]
    fn su2_projection_is_orthogonal_and_uses_categorical_dimension() {
        let space = su2_multitree_space();
        let shape = vec![2, 2, 2, 2];
        let x = (0..16)
            .map(|index| (index as f64 - 7.0) / 9.0)
            .collect::<Vec<_>>();
        let y = (0..16)
            .map(|index| (5.0 - index as f64) / 11.0)
            .collect::<Vec<_>>();

        let px_reduced = project_physical_host(&space, &shape, &x).unwrap();
        let py_reduced = project_physical_host(&space, &shape, &y).unwrap();
        let (_, px) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&space, &px_reduced).unwrap())
                .unwrap();
        let (_, py) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&space, &py_reduced).unwrap())
                .unwrap();
        let ppx = project_physical_host(&space, &shape, &px).unwrap();
        assert_close(&ppx, &px_reduced);

        let x_py = x.iter().zip(&py).map(|(&a, &b)| a * b).sum::<f64>();
        let px_y = px.iter().zip(&y).map(|(&a, &b)| a * b).sum::<f64>();
        assert!((x_py - px_y).abs() < 2.0e-12);

        // Every block couples to j=1/2, whose categorical dimension is 2.
        // The physical embedding therefore has norm² = dim(c) times the
        // reduced Euclidean norm²; this pins projection's inv_dim_scalar(c).
        let physical_norm = px.iter().map(|value| value * value).sum::<f64>();
        let reduced_norm = px_reduced.iter().map(|value| value * value).sum::<f64>();
        assert!((physical_norm - 2.0 * reduced_norm).abs() < 2.0e-12);
    }

    #[test]
    fn u1_non_self_dual_z_matches_tensorkit_formula() {
        let plus = U1Irrep::new(1);
        let minus = U1Irrep::new(-1);
        let dual_plus = SectorLeg::new([(plus, 1)], true);
        let minus_leg = SectorLeg::new([(minus, 1)], false);
        let space = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(U1FusionRule),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([dual_plus, minus_leg]),
                FusionProductSpace::new([]),
            ),
        )
        .unwrap();
        let reduced = [3.25];
        let (shape, physical) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&space, &reduced).unwrap())
                .unwrap();

        // TensorKit: Z_q^† = conj(sqrt(dim(q)) C[dual(q),q,1]). For U(1)
        // every factor is exactly one, including the non-self-dual q=+1 leg.
        assert_eq!(shape, [1, 1]);
        assert_eq!(physical, reduced);
        assert_eq!(
            project_physical_host(&space, &shape, &physical).unwrap(),
            reduced
        );
    }

    #[test]
    fn rank_zero_and_disjoint_dense_data_are_handled() {
        let scalar_space = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(U1FusionRule),
            FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
        )
        .unwrap();
        let reduced = [2.5];
        let (shape, physical) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&scalar_space, &reduced).unwrap())
                .unwrap();
        assert!(shape.is_empty());
        assert_eq!(physical, reduced);
        assert_eq!(
            project_physical_host(&scalar_space, &shape, &physical).unwrap(),
            reduced
        );

        let plus = SectorLeg::new([(U1Irrep::new(1), 1)], false);
        let minus = SectorLeg::new([(U1Irrep::new(-1), 1)], false);
        let disjoint = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(U1FusionRule),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([plus]),
                FusionProductSpace::new([minus]),
            ),
        )
        .unwrap();
        assert_eq!(disjoint.space().required_len().unwrap(), 0);
        assert!(project_physical_host(&disjoint, &[1, 1], &[7.0])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn projection_preserves_exact_strided_layout_and_zeroes_holes() {
        let rule = U1FusionRule;
        let leg = || SectorLeg::new([(U1Irrep::new(0), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        );
        let key = homspace.fusion_tree_keys(&rule)[0].clone();
        let structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
            key.into(),
            vec![1, 1],
            vec![1, 1],
            3,
        )
        .unwrap()])
        .unwrap();
        let layout = DynamicFusionMapSpace::try_from_layout(homspace, structure)
            .unwrap()
            .try_bind_rule(&rule)
            .unwrap();
        let raw = layout;
        let bound =
            BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::new(rule)).unwrap();

        let projected = project_physical_host(&bound, &[1, 1], &[4.5]).unwrap();
        assert_eq!(projected.len(), bound.space().required_len().unwrap());
        assert_eq!(projected, [0.0, 0.0, 0.0, 4.5]);
        let (_, expanded) =
            expand_physical_host(BoundDynamicTensorRef::try_new(&bound, &projected).unwrap())
                .unwrap();
        assert_eq!(expanded, [4.5]);
    }

    #[test]
    fn physical_shape_and_length_fail_before_output_publication() {
        let space = su2_multitree_space();
        assert!(matches!(
            project_physical_host(&space, &[16], &[0.0; 16]),
            Err(PhysicalConversionError::Operation(
                OperationError::ShapeMismatch { .. }
            ))
        ));
        assert!(matches!(
            project_physical_host(&space, &[2, 2, 2, 2], &[0.0; 15]),
            Err(PhysicalConversionError::Operation(
                OperationError::ElementCountMismatch { .. }
            ))
        ));
        assert!(matches!(
            checked_product::<_, std::convert::Infallible>([usize::MAX, 2]),
            Err(PhysicalConversionError::Operation(
                OperationError::ElementCountOverflow
            ))
        ));
    }

    #[test]
    fn provider_failure_remains_typed_during_transactional_staging() {
        let space = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(FailingPhysicalU1 {
                calls: Arc::new(AtomicUsize::new(0)),
                fail: true,
            }),
            FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([])),
        )
        .unwrap();
        assert!(matches!(
            project_physical_host(&space, &[], &[1.0]),
            Err(PhysicalConversionError::Provider(ProbePhysicalError))
        ));
    }
}
