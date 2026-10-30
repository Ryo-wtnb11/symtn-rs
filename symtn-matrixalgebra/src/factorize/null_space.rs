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
        lease,
        source,
        Some(FactorFamily::Null),
        |space, spectrum| null_diagonal(&M::authority(space), space, spectrum, side),
        |dense, input| null_dense(engine, dense, &M::authority(input.space()), input, side),
    )
}

/// Left null space of `source` in fusion mode `M`: per coupled sector the
/// `rows - cols` (when positive) trailing columns of the full QR's `Q`
/// (MatrixAlgebraKit `qr_null!`). The null dimension follows from the space
/// alone, so a checked dense input runs every QR before the bond is
/// published; a compact diagonal has an empty null space.
#[doc(hidden)]
pub fn left_null_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    null_from_source::<M, _, _, _, _>(engine, lease, source, FactorSide::Left)
}

/// Right null space of `source` in fusion mode `M`: per coupled sector the
/// `cols - rows` (when positive) trailing rows of the full LQ's `Q`
/// (MatrixAlgebraKit `lq_null!`); see [`left_null_from_source`].
#[doc(hidden)]
pub fn right_null_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<BoundDynFactor<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    null_from_source::<M, _, _, _, _>(engine, lease, source, FactorSide::Right)
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
    let mut pairs = Vec::new();
    in_linalg_scope(engine, dense, |dense, engine| {
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let (rows, cols) = (matrix.rows, matrix.cols);
            let (extent, other) = match side {
                FactorSide::Left => (rows, cols),
                FactorSide::Right => (cols, rows),
            };
            if extent <= other {
                null_dimensions.remove(&matrix.sector);
                continue;
            }
            let null_dim = extent - other;
            let basis = qr_null_basis(engine, dense, matrix.data, rows, cols, side)?;
            null_dimensions.insert(matrix.sector, null_dim);
            let (left, right) = match side {
                FactorSide::Left => (basis, Vec::new()),
                FactorSide::Right => (Vec::new(), adjoint_col_major(&basis, cols, null_dim)),
            };
            pairs.push(FactorPair {
                sector: matrix.sector,
                kept: null_dim,
                left,
                left_rows: rows,
                right,
                right_leading: null_dim,
            });
        }
        Ok(())
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

/// The orthonormal complement `N` (`m x (m - n)`, column-major) of the
/// column space of the `rows x cols` column-major `matrix` (`Left`, `m =
/// rows`) or of its adjoint (`Right`, `m = cols`), for `m > n`: the last
/// `m - n` columns of [`augmented_identity_qr`]'s `Q`. This is
/// MatrixAlgebraKit `qr_null!`'s `Q * [0; I]` up to a unitary gauge on the
/// null block, which the null-space contract leaves free.
fn qr_null_basis<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    matrix: &[D],
    rows: usize,
    cols: usize,
    side: FactorSide,
) -> Result<Vec<D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let (m, n) = match side {
        FactorSide::Left => (rows, cols),
        FactorSide::Right => (cols, rows),
    };
    let (mut q, _) = augmented_identity_qr(engine, dense, m, n, |block| match side {
        FactorSide::Left => block.copy_from_slice(matrix),
        FactorSide::Right => adjoint_col_major_into(matrix, rows, cols, block),
    })?;
    q.drain(..m * n);
    Ok(q)
}
