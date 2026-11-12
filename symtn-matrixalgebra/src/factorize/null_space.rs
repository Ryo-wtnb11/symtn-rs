use super::*;
use symtn_dense::Engine;

/// Null space of a compact diagonal, published through `authority`.
///
/// A diagonal sector is square, so its shape-based nullity is zero and every
/// sector drops out of `W`: the empty null space of MatrixAlgebraKit's
/// `qr_null!`/`lq_null!` with `DiagonalAlgorithm`
/// (`src/implementations/{qr,lq}.jl`, `_diagonal_qr_null!`), whatever the
/// values. The bond is still admitted so a nonfinite or malformed input is
/// refused as on the dense route.
fn null_diagonal<A, R, D>(
    authority: &A,
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<OperationError>,
    D: FactorScalar,
{
    let bond = diagonal_bond(authority, space, spectrum, FactorFamily::Null)?;
    let homspace = bond.space().space().homspace();
    let mut null_dimensions = authority.coupled_dimensions(match side {
        FactorSide::Left => homspace.codomain(),
        FactorSide::Right => homspace.domain(),
    })?;
    for region in bond.iter() {
        null_dimensions.remove(&region.coupled());
    }
    publish_one_sided_factor(
        authority,
        homspace,
        &bond.regions,
        &mut Vec::new(),
        &null_dimensions,
        side,
        FactorPlacement::Direct,
    )
}

fn null_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    factor_from_source(
        engine,
        lease,
        source,
        Some(FactorFamily::Null),
        |space, spectrum| null_diagonal(&M::authority(space), space, spectrum, side),
        |engine, dense, input| null_dense(engine, dense, &M::authority(input.space()), input, side),
        no_adjoint(),
    )
}

impl<R, D: FactorScalar> FactorOp<R, D> for op::LeftNull {
    type Output = BoundDynFactor<R, D>;
}

/// Left null space in fusion mode `M`: per coupled sector the
/// `rows - cols` (when positive) trailing columns of the full QR's `Q`
/// (MatrixAlgebraKit `qr_null!`). The null dimension follows from the space
/// alone, so a checked dense input runs every QR before the bond is
/// published; a compact diagonal has an empty null space. A lazy adjoint is
/// refused; the facade redirects it to the opposite null space.
impl<R, D: FactorScalar> Run<R, D> for op::LeftNull {
    fn run<M, L>(
        self,
        _: Token,
        engine: &mut dyn Engine,
        lease: L,
        source: FactorSource<'_, R, D>,
    ) -> Result<<Self as FactorOp<R, D>>::Output, M::Error>
    where
        M: FactorMode<R>,
        L: ExecutorLease,
    {
        null_from_source::<M, _, _, _, _>(engine, lease, source, FactorSide::Left)
    }
}

impl<R, D: FactorScalar> FactorOp<R, D> for op::RightNull {
    type Output = BoundDynFactor<R, D>;
}

/// Right null space in fusion mode `M`: per coupled sector the
/// `cols - rows` (when positive) trailing rows of the full LQ's `Q`
/// (MatrixAlgebraKit `lq_null!`); see [`op::LeftNull`].
impl<R, D: FactorScalar> Run<R, D> for op::RightNull {
    fn run<M, L>(
        self,
        _: Token,
        engine: &mut dyn Engine,
        lease: L,
        source: FactorSource<'_, R, D>,
    ) -> Result<<Self as FactorOp<R, D>>::Output, M::Error>
    where
        M: FactorMode<R>,
        L: ExecutorLease,
    {
        null_from_source::<M, _, _, _, _>(engine, lease, source, FactorSide::Right)
    }
}

#[cfg(test)]
/// Left null space `N : codomain <- W` (MatrixAlgebraKit `left_null`).
///
/// Each sector keeps the `rows - cols` (when positive) trailing columns of its
/// full QR's `Q`; sectors with no null directions drop out of `W`.
pub(crate) fn left_null<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<BoundTensorMap<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = left_null_dyn(engine, dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

#[cfg(test)]
/// Provider-bound dynamic-rank [`left_null`].
pub(crate) fn left_null_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    null_dense(
        engine,
        dense,
        &MfAuthority(input.space()),
        input,
        FactorSide::Left,
    )
}

#[cfg(test)]
/// Right null space `N : W <- domain` (MatrixAlgebraKit `right_null`).
///
/// Each sector keeps the `cols - rows` (when positive) trailing rows of its
/// full LQ's `Q`; sectors with no null directions drop out of `W`.
pub(crate) fn right_null<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<BoundTensorMap<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let out = right_null_dyn(engine, dense, &input.dynamic())?;
    typed_from_bound_factor(out)
}

#[cfg(test)]
/// Provider-bound dynamic-rank [`right_null`].
pub(crate) fn right_null_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    null_dense(
        engine,
        dense,
        &MfAuthority(input.space()),
        input,
        FactorSide::Right,
    )
}

#[cfg(test)]
/// Checked-Generic left null space.
///
/// Structural dimensions are validated before dense work. All QRs are then
/// staged before the bond is admitted and scattered by the shared checked
/// factor builder.
pub(crate) fn left_null_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_dense(
        engine,
        dense,
        &CheckedAuthority(input.space().provider_arc()),
        input,
        FactorSide::Left,
    )
}

#[cfg(test)]
/// Checked-Generic right null space; see
/// [`left_null_dyn_checked_generic`] for the transaction boundary.
pub(crate) fn right_null_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    null_dense(
        engine,
        dense,
        &CheckedAuthority(input.space().provider_arc()),
        input,
        FactorSide::Right,
    )
}

/// Shape-based null space of every coupled sector of `input`, published
/// through `authority`.
///
/// Why not a rank-revealing SVD: the null dimension of MatrixAlgebraKit's
/// default (`LeftNullViaQR`/`RightNullViaLQ`, `src/interface/orthnull.jl`)
/// is `(rows - cols)₊` per sector, decided by the space alone, so a sector
/// whose null side is not the longer one needs no dense work at all.
fn null_dense<A, E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    authority: &A,
    input: &BoundDynamicTensorRef<'_, R, D>,
    side: FactorSide,
) -> Result<BoundDynFactor<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let space = input.space().space();
    let matrices = generic_input_matricizations(space.structure(), input.data(), space.nout())?;
    // A sector only on the null side has no tensor block but is entirely null.
    let mut null_dimensions = authority.coupled_dimensions(match side {
        FactorSide::Left => space.homspace().codomain(),
        FactorSide::Right => space.homspace().domain(),
    })?;
    let mut pairs = in_linalg_scope(engine, dense, |dense, engine| {
        null_packed(engine, dense, &matrices, &mut null_dimensions, side)
    })?;
    with_input_geometry!(&matrices, |geometry| publish_one_sided_factor(
        authority,
        space.homspace(),
        geometry,
        &mut pairs,
        &null_dimensions,
        side,
        FactorPlacement::Direct,
    ))
}

/// The null bases of every sector longer on the null `side` (`rows > cols`
/// for [`FactorSide::Left`], `cols > rows` for [`FactorSide::Right`]) as one
/// solver-first executor batch ([`DenseFactorization::QrNull`]), packed
/// into one buffer in matrix order; the other sectors have no null
/// directions, are not solved, and leave `null_dimensions`. Each basis is
/// any orthonormal basis of the complement, which may differ by executor.
///
/// A left basis `N` (`rows x (rows - cols)`) is written once at its final
/// size. For the right side the batch reads each block's transpose `X^T` in
/// place (a strided view): its basis `N'` (`cols x (cols - rows)`,
/// `N'^H X^T = 0`) gives `N = N'^T`, since `X N^H = X conj(N') =
/// (N'^H X^T)^T = 0` and `N N^H = conj(N'^H N') = I`. `N'` is solved into
/// one packed scratch and transposed once into the final buffer.
fn null_packed<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrices: &(impl SectorMatrices<D> + ?Sized),
    null_dimensions: &mut BTreeMap<SectorId, usize>,
    side: FactorSide,
) -> Result<PackedFactorPairs<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    // `(long, short)`: the extent on the null side and the other one.
    let extents = |&(_, rows, cols): &(&[D], usize, usize)| match side {
        FactorSide::Left => (rows, cols),
        FactorSide::Right => (cols, rows),
    };
    let (sectors, blocks) = sector_blocks(matrices)?;
    let (sectors, blocks): (Vec<_>, Vec<_>) = sectors
        .into_iter()
        .zip(blocks)
        .filter(|(sector, block)| {
            let (long, short) = extents(block);
            if long > short {
                null_dimensions.insert(*sector, long - short);
                true
            } else {
                null_dimensions.remove(sector);
                false
            }
        })
        .unzip();
    let ranges = packed_ranges(blocks.iter().map(|block| {
        let (long, short) = extents(block);
        long * (long - short)
    }));
    let mut basis = vec![D::zero(); ranges.last().map_or(0, |range| range.end)];
    let mut read_layouts = Vec::new();
    let inputs = block_reads(&blocks, &mut read_layouts, side == FactorSide::Right)?;
    let layouts = blocks
        .iter()
        .map(|block| {
            let (long, short) = extents(block);
            [
                matrix_layout(long, long - short),
                vector_layout(0),
                vector_layout(0),
            ]
        })
        .collect::<Vec<DestLayouts>>();
    let mut dests = split_ranges(&mut basis, &ranges)?
        .into_iter()
        .zip(&layouts)
        .map(|(basis, layout)| {
            Ok(FactorDest::new(
                [Some(dest_write(basis, &layout[0])?), None, None],
                symtn_dense::FactorPlacement::AsComputed,
            ))
        })
        .collect::<Result<Vec<_>, OperationError>>()?;
    factorize_into_solver_first(
        engine,
        dense,
        DenseFactorization::QrNull,
        &inputs,
        &mut dests,
        |_, _| Ok(()),
    )?;
    let mut pairs = Vec::with_capacity(blocks.len());
    for ((block, sector), range) in blocks.iter().zip(sectors).zip(ranges) {
        let (long, short) = extents(block);
        let null = long - short;
        let (left, right) = match side {
            FactorSide::Left => (range, 0..0),
            FactorSide::Right => (0..0, range),
        };
        pairs.push(PackedFactorPair {
            sector,
            kept: null,
            left,
            left_rows: block.1,
            right,
            right_leading: null,
        });
    }
    if side == FactorSide::Left {
        return Ok(PackedFactorPairs {
            pairs,
            left: basis,
            right: Vec::new(),
        });
    }
    let mut right = vec![D::zero(); basis.len()];
    for (block, pair) in blocks.iter().zip(&pairs) {
        let (long, short) = extents(block);
        let range = pair.right.clone();
        transpose_col_major_into(
            &basis[range.clone()],
            long,
            long - short,
            &mut right[range],
            |value| value,
        );
    }
    Ok(PackedFactorPairs {
        pairs,
        left: Vec::new(),
        right,
    })
}
