use super::*;

/// The provider-labelled identity of one stored block: the fusion tree on each
/// side of the tensor map, decoded through the codec — the labelled
/// counterpart of [`crate::typed::FusionTreePairKey`], named after TensorKit's
/// `fusiontrees(t)`.
///
/// Why not `BlockSectors` / `block_sectors`: TensorKit's `blocksectors(t)` is
/// the set of coupled sectors of a tensor, which is a strictly smaller thing
/// than a per-block tree pair. Reusing the name for something else would be a
/// false friend for anyone arriving from TensorKit.
///
/// Inner lines are part of the identity, not decoration: from rank three up,
/// two distinct blocks can share their uncoupled and coupled sectors and
/// differ only in how the intermediate fusions ran, so a key without them
/// would not name a block.
///
/// Vertex labels are part of the identity for Generic providers: two trees
/// can have identical sector labels and differ only by outer multiplicity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockFusionTrees<S> {
    pub(super) coupled: S,
    pub(super) codomain_uncoupled: Vec<S>,
    pub(super) codomain_innerlines: Vec<S>,
    pub(super) codomain_vertices: Vec<MultiplicityIndex>,
    pub(super) domain_uncoupled: Vec<S>,
    pub(super) domain_innerlines: Vec<S>,
    pub(super) domain_vertices: Vec<MultiplicityIndex>,
}

impl<S> BlockFusionTrees<S> {
    /// The sector both trees couple to.
    #[inline]
    pub fn coupled(&self) -> &S {
        &self.coupled
    }

    /// Codomain leg sectors, in codomain axis order.
    #[inline]
    pub fn codomain_uncoupled(&self) -> &[S] {
        &self.codomain_uncoupled
    }

    /// Codomain intermediate fusion sectors, from the innermost outwards.
    #[inline]
    pub fn codomain_innerlines(&self) -> &[S] {
        &self.codomain_innerlines
    }

    /// Codomain outer-multiplicity labels, in fusion-vertex order.
    #[inline]
    pub fn codomain_vertices(&self) -> &[MultiplicityIndex] {
        &self.codomain_vertices
    }

    /// Domain leg sectors, in domain axis order.
    ///
    /// These are the domain spaces' own sectors (TensorKit's `f2.uncoupled`),
    /// not their duals; on both sides the uncoupled sectors fuse to
    /// [`Self::coupled`].
    #[inline]
    pub fn domain_uncoupled(&self) -> &[S] {
        &self.domain_uncoupled
    }

    /// Domain intermediate fusion sectors, from the innermost outwards.
    #[inline]
    pub fn domain_innerlines(&self) -> &[S] {
        &self.domain_innerlines
    }

    /// Domain outer-multiplicity labels, in fusion-vertex order.
    #[inline]
    pub fn domain_vertices(&self) -> &[MultiplicityIndex] {
        &self.domain_vertices
    }
}

/// One coupled-sector matrix of a tensor map, borrowed without a copy:
/// TensorKit's `block(t, c)`. See [`TensorMap::block`].
///
/// Rows are the codomain fusion trees of the sector and columns the domain
/// trees. Each tree owns a contiguous range of rows (columns) whose length is
/// the product of its legs' degeneracies, indexed column-major in those legs.
/// The trees and their ranges, in matrix order, are [`Self::row_trees`] and
/// [`Self::col_trees`]. That order is symtn's storage order, which is in
/// general a permutation of TensorKit's `fusiontrees` order (see
/// `docs/sector_id_compatibility.md`); presenting TensorKit's order would copy.
#[derive(Debug)]
pub struct CoupledBlock<'a, R, D, S = Vec<D>> {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) payload: CoupledBlockPayload<'a, D, S>,
    pub(super) provider: &'a R,
    /// The stored region and whether rows and columns are swapped relative
    /// to it (a lazy adjoint); `None` for an absent sector's empty view.
    pub(super) region: Option<(Arc<[CoupledSectorRegion]>, usize, bool)>,
}

/// Where the entries of a [`CoupledBlock`] live.
#[derive(Debug)]
#[non_exhaustive]
pub enum CoupledBlockPayload<'a, D, S = Vec<D>> {
    /// A column-major matrix inside `storage`, starting at `offset`.
    ///
    /// Without `adjoint` it is the block itself, with leading dimension
    /// `rows`. With `adjoint` it is the stored `cols × rows` matrix of a lazy
    /// adjoint's parent, with leading dimension `cols`, and the block is its
    /// conjugate transpose (the GEMM operand flag `Adjoint`).
    Dense {
        /// The tensor's canonical payload (the parent's, for a lazy adjoint).
        storage: &'a S,
        /// Element offset of the stored matrix's first entry.
        offset: usize,
        /// Whether the block is the conjugate transpose of the stored matrix.
        adjoint: bool,
    },
    /// A compact diagonal: the block is `diag(values)`.
    Diagonal(&'a [D]),
}

impl<D, S> Clone for CoupledBlockPayload<'_, D, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<D, S> Copy for CoupledBlockPayload<'_, D, S> {}

impl<R, D, S> Clone for CoupledBlock<'_, R, D, S> {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            payload: self.payload,
            provider: self.provider,
            region: self.region.clone(),
        }
    }
}

/// The provider labels of one fusion tree: one side of a
/// [`BlockFusionTrees`] pair, as [`CoupledBlock::row_trees`] and
/// [`CoupledBlock::col_trees`] report it.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FusionTreeLabels<S> {
    coupled: S,
    uncoupled: Vec<S>,
    innerlines: Vec<S>,
    vertices: Vec<MultiplicityIndex>,
}

impl<S> FusionTreeLabels<S> {
    /// The sector the tree couples to.
    #[inline]
    pub fn coupled(&self) -> &S {
        &self.coupled
    }

    /// Leg sectors, in axis order.
    #[inline]
    pub fn uncoupled(&self) -> &[S] {
        &self.uncoupled
    }

    /// Intermediate fusion sectors, from the innermost outwards.
    #[inline]
    pub fn innerlines(&self) -> &[S] {
        &self.innerlines
    }

    /// Outer-multiplicity labels, in fusion-vertex order.
    #[inline]
    pub fn vertices(&self) -> &[MultiplicityIndex] {
        &self.vertices
    }
}

/// Decoded trees with their row or column ranges, in matrix order.
pub(super) type TreeExtents<S> = Vec<(FusionTreeLabels<S>, core::ops::Range<usize>)>;

impl<'a, R, D, S> CoupledBlock<'a, R, D, S> {
    /// Number of rows: the codomain block dimension of the sector.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns: the domain block dimension of the sector.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The borrowed payload, for callers that hand the region to a kernel.
    #[inline]
    pub fn payload(&self) -> CoupledBlockPayload<'a, D, S> {
        self.payload
    }
}

impl<R, D, S> CoupledBlock<'_, R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    /// The codomain fusion trees in row order, each with its row range.
    ///
    /// Decodes labels and allocates the returned list; reads no payload.
    pub fn row_trees(&self) -> Result<TreeExtents<R::Sector>, FacadeError<R>> {
        self.trees(false)
    }

    /// The domain fusion trees in column order, each with its column range.
    ///
    /// Decodes labels and allocates the returned list; reads no payload.
    pub fn col_trees(&self) -> Result<TreeExtents<R::Sector>, FacadeError<R>> {
        self.trees(true)
    }

    fn trees(&self, columns: bool) -> Result<TreeExtents<R::Sector>, FacadeError<R>> {
        let Some((regions, index, swapped)) = &self.region else {
            return Ok(Vec::new());
        };
        let region = &regions[*index];
        let extents = if columns != *swapped {
            region.col_trees()
        } else {
            region.row_trees()
        };
        extents
            .iter()
            .map(|extent| {
                let tree = extent.tree();
                let labels = FusionTreeLabels {
                    coupled: TypedSectorAdmission::try_decode_label(self.provider, tree.coupled())
                        .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
                    uncoupled: decode_sectors(self.provider, tree.uncoupled())?,
                    innerlines: decode_sectors(self.provider, tree.innerlines())?,
                    vertices: tree.vertices().to_vec(),
                };
                let end = extent
                    .offset()
                    .checked_add(extent.extent().map_err(Error::from)?)
                    .ok_or_else(|| internal_layout_error("tree extent overflow"))?;
                Ok((labels, extent.offset()..end))
            })
            .collect()
    }
}

impl<R, D, S> CoupledBlock<'_, R, D, S>
where
    D: TensorScalar,
    S: HostReadableStorage<D>,
{
    /// Entry `(row, col)` of the block, or `None` outside `rows × cols`.
    pub fn get(&self, row: usize, col: usize) -> Option<D> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        Some(match self.payload {
            CoupledBlockPayload::Dense {
                storage,
                offset,
                adjoint: false,
            } => storage.as_slice()[offset + row + col * self.rows],
            CoupledBlockPayload::Dense {
                storage,
                offset,
                adjoint: true,
            } => FactorScalar::adjoint(storage.as_slice()[offset + col + row * self.cols]),
            CoupledBlockPayload::Diagonal(values) if row == col => values[row],
            CoupledBlockPayload::Diagonal(_) => D::from_real(0.0),
        })
    }
}

pub(super) fn decode_sectors<R>(
    provider: &R,
    ids: &[symtn_core::SectorId],
) -> Result<Vec<R::Sector>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    ids.iter()
        .map(|&id| {
            TypedSectorAdmission::try_decode_label(provider, id)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
        })
        .collect()
}

/// Decodes one block key into provider labels.
///
/// Every id here came out of the engine's own fusion enumeration, so a failure
/// is the provider breaking [`SectorCodec`]'s decode-totality law, and it is
/// surfaced as the codec's own error rather than a panic.
pub(super) fn decode_block_fusion_trees<R>(
    provider: &R,
    key: &BlockKey,
) -> Result<BlockFusionTrees<R::Sector>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    let pair = key.as_fusion_tree_pair().ok_or_else(|| {
        FacadeError::<R>::from(Error::InvalidArgument(format!(
            "block key is {}, not a fusion-tree pair",
            key.kind()
        )))
    })?;
    let codomain = pair.codomain_tree();
    let domain = pair.domain_tree();
    Ok(BlockFusionTrees {
        coupled: TypedSectorAdmission::try_decode_label(provider, codomain.coupled())
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?,
        codomain_uncoupled: decode_sectors(provider, codomain.uncoupled())?,
        codomain_innerlines: decode_sectors(provider, codomain.innerlines())?,
        codomain_vertices: codomain.vertices().to_vec(),
        domain_uncoupled: decode_sectors(provider, domain.uncoupled())?,
        domain_innerlines: decode_sectors(provider, domain.innerlines())?,
        domain_vertices: domain.vertices().to_vec(),
    })
}

pub(super) fn map_block_fusion_trees<A, B>(
    source: &BlockFusionTrees<A>,
    map: impl Fn(&A) -> B,
) -> BlockFusionTrees<B> {
    BlockFusionTrees {
        coupled: map(&source.coupled),
        codomain_uncoupled: source.codomain_uncoupled.iter().map(&map).collect(),
        codomain_innerlines: source.codomain_innerlines.iter().map(&map).collect(),
        codomain_vertices: source.codomain_vertices.clone(),
        domain_uncoupled: source.domain_uncoupled.iter().map(&map).collect(),
        domain_innerlines: source.domain_innerlines.iter().map(map).collect(),
        domain_vertices: source.domain_vertices.clone(),
    }
}

pub(super) struct PreparedProductOperand<'a, S, P, D>
where
    P: SectorCodec,
{
    source: &'a TensorMap<S, D>,
    codomain: Vec<GradedSpace<P>>,
    domain: Vec<GradedSpace<P>>,
    #[expect(
        clippy::type_complexity,
        reason = "each prepared block stores its source index and two axis maps together"
    )]
    blocks: HashMap<BlockFusionTrees<P::Sector>, (usize, Vec<usize>, Vec<usize>)>,
}

pub(super) fn prepare_product_operand<S, P, D>(
    source: &TensorMap<S, D>,
    provider: Arc<P>,
    embed: impl Fn(<S as SectorCodec>::Sector) -> <P as SectorCodec>::Sector,
    project: impl Fn(&<P as SectorCodec>::Sector) -> <S as SectorCodec>::Sector,
) -> Result<PreparedProductOperand<'_, S, P, D>, Error>
where
    S: TypedSectorAdmission<
            Sector = <S as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    P: TypedSectorAdmission<
            Sector = <P as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    D: TensorScalar,
{
    let embed_legs = |legs: Vec<GradedSpace<S>>| -> Result<Vec<GradedSpace<P>>, Error> {
        legs.into_iter()
            .map(|leg| {
                let sectors = leg.sectors()?;
                GradedSpace::try_new(
                    Arc::clone(&provider),
                    sectors
                        .into_iter()
                        .zip(leg.degeneracies().iter().copied())
                        .map(|(sector, degeneracy)| (embed(sector), degeneracy)),
                )
                .and_then(|space| {
                    if leg.is_dual() {
                        space.try_dual()
                    } else {
                        Ok(space)
                    }
                })
            })
            .collect()
    };
    let codomain = embed_legs(source.codomain())?;
    let domain = embed_legs(source.domain())?;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new(codomain.iter().map(|leg| leg.leg().clone())),
        FusionProductSpace::new(domain.iter().map(|leg| leg.leg().clone())),
    );
    let mut blocks = HashMap::with_capacity(source.subblock_count());
    for index in 0..source.subblock_count() {
        let key = source.subblock_fusion_trees(index)?;
        let block = source.subblock(index)?;
        blocks.insert(
            key,
            (
                block.offset(),
                block.shape().to_vec(),
                block.strides().to_vec(),
            ),
        );
    }
    // Stage (do not publish) the product layout and prove that canonical-unit
    // projection is a bijection onto the source blocks. The callback below can
    // therefore not discover `missing_block` only after target admission.
    let prepared = homspace.prepare_fusion_tree_layout_checked(provider.as_ref())?;
    let mut projected = HashSet::with_capacity(prepared.keys().len());
    let mut target_blocks = HashMap::with_capacity(prepared.keys().len());
    for key in prepared.keys() {
        let labelled =
            decode_block_fusion_trees(provider.as_ref(), &BlockKey::FusionTree(key.clone()))?;
        let source_key = map_block_fusion_trees(&labelled, &project);
        let Some(layout) = blocks.get(&source_key).cloned() else {
            return Err(Error::InvalidArgument(
                "canonical-unit product embedding did not preserve source blocks".to_string(),
            ));
        };
        if !projected.insert(source_key) || target_blocks.insert(labelled, layout).is_some() {
            return Err(Error::InvalidArgument(
                "canonical-unit product embedding did not preserve source blocks".to_string(),
            ));
        }
    }
    if projected.len() != blocks.len() {
        return Err(Error::InvalidArgument(
            "canonical-unit product embedding did not preserve source blocks".to_string(),
        ));
    }

    Ok(PreparedProductOperand {
        source,
        codomain,
        domain,
        blocks: target_blocks,
    })
}

impl<S, P, D> PreparedProductOperand<'_, S, P, D>
where
    S: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    P: TypedSectorAdmission<
            Sector = <P as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    D: TensorScalar,
{
    pub(super) fn commit(self) -> Result<TensorMap<P, D>, Error> {
        let source = self.source;
        let materialized = source.materialized_tensor_uncached()?;
        let data_payload = materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data();
        let data: &[D] = &data_payload;
        let blocks = self.blocks;
        let codomain = self.codomain;
        let domain = self.domain;
        let mut missing_block = false;
        let built = TensorMap::from_subblock_fn(
            source.runtime(),
            codomain.iter(),
            domain.iter(),
            |key, indices| {
                let Some((offset, shape, strides)) = blocks.get(key) else {
                    missing_block = true;
                    return D::from_real(0.0);
                };
                if indices.len() != shape.len()
                    || indices.iter().zip(shape).any(|(&index, &dim)| index >= dim)
                {
                    missing_block = true;
                    return D::from_real(0.0);
                }
                let position = indices
                    .iter()
                    .zip(strides)
                    .fold(*offset, |position, (&index, &stride)| {
                        position + index * stride
                    });
                data[position]
            },
        )?;
        if missing_block {
            return Err(Error::InvalidArgument(
                "canonical-unit product embedding did not preserve a source block".to_string(),
            ));
        }
        Ok(built)
    }
}

/// One coupled sector's factorization spectrum, labelled through the provider:
/// the typed counterpart of [`symtn_matrixalgebra::SectorSpectrum`], whose
/// `sector` is a raw [`crate::sector::SectorId`].
///
/// Why decode rather than extend the raw-id exception that [`TensorMap::subblock`]
/// carries: that exception is scoped to engine layout views, and a spectrum is
/// caller-facing physics — [`TensorMap::svd_vals`]'s entire return would
/// otherwise be raw ids.
///
/// `values` is descending by magnitude, as the seam guarantees.
///
/// `V` is the value type and defaults to [`f64`], the singular/Hermitian-
/// eigenvalue case, so `SectorSpectrum<S>` keeps its meaning. The general
/// eigendecompositions spell it `SectorSpectrum<S, Complex64>`. The default is
/// what [`symtn_matrixalgebra::SectorSpectrum`] already does, for the same
/// reason: a real spectrum is by far the common one, and a caller who never
/// touches `eig_*` should never have to name the parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct SectorSpectrum<S, V = f64> {
    /// The coupled sector, in the provider's own labels.
    pub sector: S,
    /// That sector's values. Public diagonal construction preserves this order;
    /// factorization outputs separately use descending magnitude order.
    pub values: Vec<V>,
}

/// The two block payload representations one typed tensor map can carry.
///
/// `D` is a type parameter, so one diagonal arm holds values of exactly the
/// payload type.
pub(super) enum TypedData<D, S = Vec<D>> {
    /// The dense coupled-sector buffer every operation can read.
    Dense(S),
    /// Compact O(Σ_c k_c) storage for a proved bond endomorphism, including
    /// SVD `s`, `eigh`/`eig` `d`, and diagonal QR/LQ/polar factors: only the
    /// per-sector diagonal values, keyed by the engine's raw
    /// [`crate::sector::SectorId`] — a stored payload never leaves this module, so
    /// there is nothing here for the codec to label.
    Diagonal(Vec<symtn_matrixalgebra::SectorSpectrum<D>>),
}

/// Applies a scalar function to every stored value of a compact spectrum,
/// leaving the sector keys and the per-sector lengths untouched.
///
/// This is the whole of the O(rank) arm shared by [`TensorMap::exp`],
/// [`TensorMap::inv`] and [`TensorMap::map_diagonal`] (the pseudo-inverse's
/// is matrix-algebra `pinv_diagonal_spectrum`, which owns its cutoff): a spectral
/// function acts on eigenvalues, so it never moves weight between sectors and
/// never changes a bond dimension, which is exactly why the result can stay on
/// the space it was called on.
pub(super) fn map_spectrum<D: Copy>(
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    mut value_of: impl FnMut(D) -> Result<D, Error>,
) -> Result<Vec<symtn_matrixalgebra::SectorSpectrum<D>>, Error> {
    spectrum
        .iter()
        .map(|entry| {
            Ok(symtn_matrixalgebra::SectorSpectrum {
                sector: entry.sector,
                values: entry
                    .values
                    .iter()
                    .map(|&value| value_of(value))
                    .collect::<Result<_, Error>>()?,
            })
        })
        .collect()
}

/// TensorKit `exp(::DiagonalTensorMap)`, `exp.(d.data)`: the one compact
/// `exp` value map every fusion mode publishes. Nonfinite entries map through
/// `exp` like any other value; nothing is rejected.
pub(super) fn exp_spectrum<D: TensorScalar>(
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
) -> Result<Vec<symtn_matrixalgebra::SectorSpectrum<D>>, Error> {
    map_spectrum(spectrum, |value| Ok(value.exp_value()))
}

/// TensorKit `inv(::DiagonalTensorMap)`, `inv.(d.data)`: the one compact
/// `inv` value map every fusion mode publishes.
///
/// Why `== 0` and not a tolerance: the dense arm has none either (the solve
/// either fails or it does not), and a compact arm that refused near-zero
/// entries would let storage change the answer. Exact zero is singular and
/// reported as the dense arm's singular-solve failure,
/// `DenseError::NumericalFailure`; NaN and infinity map through the
/// reciprocal without an error.
pub(super) fn inv_spectrum<D: TensorScalar>(
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
) -> Result<Vec<symtn_matrixalgebra::SectorSpectrum<D>>, Error> {
    map_spectrum(spectrum, |value| {
        if value.abs_value() == 0.0 {
            Err(Error::from(symtn_tensors::OperationError::Dense(
                symtn_dense::DenseError::NumericalFailure {
                    backend: symtn_dense::DenseBackend::Tenferro,
                    op: "solve_into",
                    message: "inv of a singular diagonal (zero entry)".to_string(),
                },
            )))
        } else {
            Ok(value.recip_value())
        }
    })
}

/// The compact `solve` divisor's singularity check, shared by every fusion
/// mode: an exact zero is reported as the dense solve's singular-block
/// failure, so a compact and a dense divisor agree on the error variant.
pub(super) fn reject_singular_compact_divisor<D: TensorScalar>(
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
) -> Result<(), Error> {
    if spectrum
        .iter()
        .flat_map(|entry| &entry.values)
        .any(|value| value.abs_value() == 0.0)
    {
        return Err(Error::from(symtn_tensors::OperationError::Dense(
            symtn_dense::DenseError::NumericalFailure {
                backend: symtn_dense::DenseBackend::Tenferro,
                op: "solve_into",
                message: "singular compact diagonal divisor".to_string(),
            },
        )));
    }
    Ok(())
}

/// Two compact spectra that live on one bond space must agree sector for
/// sector and length for length; when they do not, the space and the payload
/// have gone out of step, which is an engine invariant break rather than
/// anything a caller did.
pub(super) fn spectra_disagree() -> Error {
    Error::InvalidArgument("equal bond spaces carry incompatible compact spectra".to_string())
}

/// Test helper: a compact diagonal factor on a fresh multiplicity-free bond
/// derived from `authority`, whose payload type may differ from the tensor it
/// came from (the factorizations build theirs from `seam::spectrum_bond`
/// and [`diagonal_factor_on_bound`]).
///
/// Borrowed rather than consumed, and filled from a slice rather than by
/// `into_iter().collect()`: the standard library's in-place collect reuses the
/// source buffer only when `E` and `V` share size and alignment, so consuming
/// the spectrum made the number of buffers a spectrum factor costs depend on
/// the payload dtype — free for `f64`, one `Vec<E>` per coupled sector for
/// `f32`, `Complex32` and `Complex64` (#1337).
#[cfg(test)]
pub(super) fn diagonal_factor_on<R, E, V>(
    runtime: &Runtime,
    authority: &BoundDynamicFusionMapSpace<R>,
    spectrum: &mut [symtn_matrixalgebra::SectorSpectrum<V>],
    to_scalar: impl Fn(V) -> E,
) -> Result<TensorMap<R, E>, Error>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    V: Copy,
{
    let space = symtn_matrixalgebra::seam::spectrum_bond::<MultiplicityFreeAdmissionMode, _, _>(
        authority, spectrum,
    )?;
    Ok(diagonal_factor_on_bound(
        runtime, space, spectrum, to_scalar,
    ))
}

pub(super) fn diagonal_factor_on_bound<R, E, V>(
    runtime: &Runtime,
    space: BoundDynamicFusionMapSpace<R>,
    spectrum: &mut [symtn_matrixalgebra::SectorSpectrum<V>],
    to_scalar: impl Fn(V) -> E,
) -> TensorMap<R, E>
where
    V: Copy,
{
    spectrum.sort_unstable_by_key(|entry| entry.sector);
    let data = spectrum
        .iter()
        .map(|entry| symtn_matrixalgebra::SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|&value| to_scalar(value)).collect(),
        })
        .collect();
    TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(space, data)),
    }
}

/// Full SVD may store `S` compactly only on the exact constructed bond.
pub(super) fn full_svd_compact_bond<R, D>(
    u: &symtn_matrixalgebra::BoundDynFactor<R, D>,
    vh: &symtn_matrixalgebra::BoundDynFactor<R, D>,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum],
) -> bool {
    let u_domain = u.space().space().homspace().domain();
    let vh_codomain = vh.space().space().homspace().codomain();
    let (Some(row), Some(col)) = (
        u_domain.legs().first().filter(|_| u_domain.len() == 1),
        vh_codomain
            .legs()
            .first()
            .filter(|_| vh_codomain.len() == 1),
    ) else {
        return false;
    };
    full_svd_spectrum_matches_bonds(row, col, spectrum)
}

pub(super) fn full_svd_spectrum_matches_bonds(
    row: &SectorLeg,
    col: &SectorLeg,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum],
) -> bool {
    let Ok(spectrum_leg) = SectorLeg::try_new(
        spectrum
            .iter()
            .map(|entry| (entry.sector, entry.values.len())),
        false,
    ) else {
        return false;
    };
    row == col && row == &spectrum_leg && spectrum.len() == row.sectors().len()
}

/// Wraps one factor the matrix-algebra seam produced into a typed tensor map
/// on `runtime`. `BoundDynFactor::into_parts` hands back exactly the pair
/// [`TypedTensorBody`] stores, so the seam's own certification suffices.
pub(super) fn wrap_factor_on<R, E>(
    runtime: &Runtime,
    factor: BoundDynFactor<R, E>,
) -> TensorMap<R, E> {
    let (space, data) = factor.into_parts();
    TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(space, data)),
    }
}

/// `dense_factor * dense + diagonal_factor * spectrum`, laid out per `space`.
///
/// The dense operand is scaled into a fresh owned buffer and the spectrum is
/// added onto that buffer's per-block diagonal, which is the only place a bond
/// space is non-zero. Same block addressing as
/// [`symtn_matrixalgebra::seam::diagonal_bond_data`], which is what put the values
/// there in the first place.
pub(super) fn scatter_spectrum<D>(
    space: &DynamicFusionMapSpace,
    dense: &[D],
    dense_factor: D,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    diagonal_factor: D,
) -> Result<Vec<D>, Error>
where
    D: TensorScalar,
{
    let mut data: Vec<D> = dense
        .iter()
        .map(|&value| scale_value(value, dense_factor))
        .collect();
    add_spectrum_into(space, &mut data, spectrum, diagonal_factor)?;
    Ok(data)
}

/// Pairs blocks with spectrum entries when both run in the same sector order.
///
/// Why a hint and not a merge that assumes sortedness: the result must not
/// depend on an ordering the type does not enforce. A block whose entry is not
/// next falls back to a scan, counted in `misses`.
#[derive(Default)]
pub(super) struct SpectrumCursor {
    next: usize,
    pub(super) misses: usize,
}

impl SpectrumCursor {
    pub(super) fn find<'a, V>(
        &mut self,
        spectrum: &'a [symtn_matrixalgebra::SectorSpectrum<V>],
        sector: symtn_core::SectorId,
    ) -> Option<&'a symtn_matrixalgebra::SectorSpectrum<V>> {
        if let Some(entry) = spectrum
            .get(self.next)
            .filter(|entry| entry.sector == sector)
        {
            self.next += 1;
            return Some(entry);
        }
        self.misses += 1;
        let position = spectrum.iter().position(|entry| entry.sector == sector)?;
        self.next = position + 1;
        Some(&spectrum[position])
    }
}

pub(super) fn add_spectrum_into<D>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    diagonal_factor: D,
) -> Result<(), Error>
where
    D: TensorScalar,
{
    add_spectrum_counting_misses(space, data, spectrum, diagonal_factor).map(drop)
}

/// [`add_spectrum_into`], returning how many blocks missed the cursor hint
/// and paid a full scan of `spectrum`.
///
/// Blocks and spectrum entries both run in ascending sector order: the bond
/// leg is sorted by `SectorLeg::build`, and every compact payload is sorted by
/// its constructor (`diagonal`, `diagonal_factor_on_bound`, the decoder). The
/// cursor therefore hits on every block and the walk is O(G). The scan on a
/// miss keeps the result independent of that ordering, so it is a cost
/// fallback and never a correctness assumption.
pub(super) fn add_spectrum_counting_misses<D>(
    space: &DynamicFusionMapSpace,
    data: &mut [D],
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    diagonal_factor: D,
) -> Result<usize, Error>
where
    D: TensorScalar,
{
    let structure = space.structure();
    let mut cursor = SpectrumCursor::default();
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let Some(pair) = block.key().as_fusion_tree_pair() else {
            continue;
        };
        let sector = pair.codomain_tree().coupled();
        let Some(entry) = cursor.find(spectrum, sector) else {
            // Both operands live on one space, and a compact payload's space is
            // built from its own spectrum, so every block's coupled sector has
            // an entry. Skipping is the safe behaviour if that ever breaks —
            // the block keeps the dense operand's contribution — but it is a
            // silent wrong answer, so say so loudly in a debug build.
            debug_assert!(
                false,
                "no spectrum entry for coupled sector {sector:?} on its own bond space"
            );
            continue;
        };
        let strides = block.strides();
        let stride = strides[0] + strides[1];
        let offset = block.offset();
        // The three lengths agree by construction, for the same reason.
        debug_assert_eq!(block.shape()[0], block.shape()[1]);
        debug_assert_eq!(block.shape()[0], entry.values.len());
        let count = block.shape()[0]
            .min(block.shape()[1])
            .min(entry.values.len());
        for (i, &value) in entry.values[..count].iter().enumerate() {
            let position = offset + i * stride;
            data[position] = data[position] + scale_value(value, diagonal_factor);
        }
    }
    debug_assert_eq!(
        cursor.misses, 0,
        "blocks and spectrum entries must both run in ascending sector order"
    );
    Ok(cursor.misses)
}

/// Whether `space` is a bond space: rank one on each side, with the same leg on
/// both — the shape a compact spectrum can address, and the only shape whose
/// dense form is block-diagonal.
///
/// This is the guard TensorKit's `DiagonalTensorMap` gets for free from its
/// type.
///
/// Applied to the *destination* of an operation — an operand's storage says
/// what it holds, only the destination says whether a compact result is
/// representable — and to a decoded compact payload's space.
///
/// # Reachability
///
/// Only decoding untrusted input can make this answer `false`. At the
/// compact-*destination* call sites it cannot fail — every
/// [`TypedData::Diagonal`] payload this module can produce sits on a space
/// admitted by [`TensorMap::diagonal`] or built by [`diagonal_factor_on`]
/// through [`symtn_matrixalgebra::seam::spectrum_bond`], which is a
/// bond space by construction. Diagonal QR/LQ preserve that exact input space,
/// and the operations that preserve the payload
/// ([`TensorMap::scale`], [`TensorMap::axpby`], [`TensorMap::adjoint`],
/// [`TensorMap::map_diagonal`], the `D * D` arm) all keep that space. It stays
/// at those sites because the next constructor of a compact payload — a
/// diagonal-aware `contract`, say — would be the first one able to aim at a
/// destination that is not a bond space, and should find the check already in
/// place rather than have to notice it is missing.
pub(super) fn is_diagonal_bond_space(space: &DynamicFusionMapSpace) -> bool {
    let homspace = space.homspace();
    space.nout() == 1 && space.nin() == 1 && homspace.codomain().legs() == homspace.domain().legs()
}
