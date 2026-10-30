use super::*;
use symtn_dense::Engine;

/// Left polar factors `W` (phase) and `P` (magnitude) of the compact diagonal
/// `spectrum` on `space`, in fusion mode `M`; both stay compact on the input
/// space.
fn left_polar_of_diagonal<M, R, D>(
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[SectorSpectrum<D>],
) -> Result<LeftPolar<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    D: FactorScalar,
{
    let (w, p) = diagonal_phase_magnitude_spectra(
        &M::authority(space),
        space,
        spectrum,
        FactorFamily::Polar,
    )?;
    let on_input = |values| FactorOutput::Diagonal {
        space: space.clone(),
        values,
    };
    Ok(LeftPolar {
        w: on_input(w),
        p: on_input(p),
    })
}

#[cfg(test)]
/// Left polar decomposition `t = W * P` (MatrixAlgebraKit `left_polar`):
/// `W` is the isometry `U * Vh` and `P = V * S * Vh` the positive part on
/// the domain. Every coupled-sector matrix must have at least as many rows as
/// columns; otherwise this returns [`OperationError::InvalidArgument`] before
/// entering the dense SVD.
pub(crate) fn left_polar<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<LeftPolar<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let LeftPolar { w, p } = left_polar_dyn(engine, dense, &input.dynamic())?;
    Ok(LeftPolar {
        w: typed_from_bound_factor(w)?,
        p: typed_from_bound_factor(p)?,
    })
}

#[cfg(test)]
/// Multiplicity-free dynamic-rank [`left_polar`].
pub(crate) fn left_polar_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LeftPolar<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let authority = MfAuthority(input.space());
    polar_dense(
        engine,
        dense,
        &authority,
        input,
        PolarDirection::Left,
        PolarDirection::Left,
    )
    .map(|(w, p)| LeftPolar { w, p })
}

#[cfg(test)]
/// Right polar decomposition `t = P * W` (MatrixAlgebraKit `right_polar`):
/// `P = U * S * U^H` is the positive part on the codomain and `W = U * Vh`.
/// Every coupled-sector matrix must have at least as many columns as rows;
/// otherwise this returns [`OperationError::InvalidArgument`] before entering
/// the dense SVD.
pub(crate) fn right_polar<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundTensorMapRef<'_, R, D>,
) -> Result<RightPolar<BoundTensorMap<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let RightPolar { p, wh } = right_polar_dyn(engine, dense, &input.dynamic())?;
    Ok(RightPolar {
        p: typed_from_bound_factor(p)?,
        wh: typed_from_bound_factor(wh)?,
    })
}

#[cfg(test)]
/// Multiplicity-free dynamic-rank [`right_polar`].
pub(crate) fn right_polar_dyn<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<RightPolar<BoundDynFactor<R, D>>, OperationError>
where
    E: DenseExecutor + ?Sized,
    R: MultiplicityFreeRigidSymbols,
    D: FactorScalar,
{
    let authority = MfAuthority(input.space());
    polar_dense(
        engine,
        dense,
        &authority,
        input,
        PolarDirection::Right,
        PolarDirection::Right,
    )
    .map(|(wh, p)| RightPolar { p, wh })
}

#[cfg(test)]
pub(crate) fn left_polar_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LeftPolar<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let authority = CheckedAuthority(input.space().provider_arc());
    polar_dense(
        engine,
        dense,
        &authority,
        input,
        PolarDirection::Left,
        PolarDirection::Left,
    )
    .map(|(w, p)| LeftPolar { w, p })
}

#[cfg(test)]
pub(crate) fn right_polar_dyn_checked_generic<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<RightPolar<BoundDynFactor<R, D>>, CheckedGenericFactorPlanError<R::Error>>
where
    E: DenseExecutor + ?Sized,
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let authority = CheckedAuthority(input.space().provider_arc());
    polar_dense(
        engine,
        dense,
        &authority,
        input,
        PolarDirection::Right,
        PolarDirection::Right,
    )
    .map(|(wh, p)| RightPolar { p, wh })
}

/// The `(W, P)` factors of one polar decomposition.
type PolarOutputs<R, D> = (FactorOutput<R, D>, FactorOutput<R, D>);

/// Polar factors of `source` in fusion mode `M`, as `(W, P)`. A compact
/// diagonal's phases and magnitudes stay compact on the input space.
fn polar_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
    direction: PolarDirection,
) -> Result<PolarOutputs<R, D>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    // The dense route checks values after its direction admission.
    factor_from_source(
        lease,
        source,
        None,
        |space, spectrum| {
            left_polar_of_diagonal::<M, R, D>(space, spectrum).map(|LeftPolar { w, p }| (w, p))
        },
        |dense, input| {
            polar_dense(
                engine,
                dense,
                &M::authority(input.space()),
                input,
                direction,
                direction,
            )
            .map(|(w, p)| (FactorOutput::Dense(w), FactorOutput::Dense(p)))
        },
    )
}

/// Left polar decomposition `W * P` of `source` in fusion mode `M`
/// (MatrixAlgebraKit `left_polar!` via `PolarViaSVD`).
#[doc(hidden)]
pub fn left_polar_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<LeftPolar<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    polar_from_source::<M, _, _, _, _>(engine, lease, source, PolarDirection::Left)
        .map(|(w, p)| LeftPolar { w, p })
}

/// Right polar decomposition `P * Wh` of `source` in fusion mode `M`
/// (MatrixAlgebraKit `right_polar!` via `PolarViaSVD`).
#[doc(hidden)]
pub fn right_polar_from_source<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
) -> Result<RightPolar<FactorOutput<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    polar_from_source::<M, _, _, _, _>(engine, lease, source, PolarDirection::Right)
        .map(|(wh, p)| RightPolar { p, wh })
}

/// Left polar of a lazy adjoint, as the dense *parent's* right polar
/// factors in fusion mode `M`: the view's left polar is `w = wh^H` and the
/// returned `p`. Direction errors name the requested left polar.
#[doc(hidden)]
pub fn left_polar_adjoint_from_parent<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<RightPolar<BoundDynFactor<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    lease
        .run(|dense| {
            polar_dense(
                engine,
                dense,
                &M::authority(parent.space()),
                parent,
                PolarDirection::Right,
                PolarDirection::Left,
            )
        })
        .map(|(wh, p)| RightPolar { p, wh })
}

/// Right polar of a lazy adjoint, as the dense *parent's* left polar factors
/// in fusion mode `M`: the view's right polar is the returned `p` and `wh =
/// w^H`. Direction errors name the requested right polar.
#[doc(hidden)]
pub fn right_polar_adjoint_from_parent<M, L, E, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    parent: &BoundDynamicTensorRef<'_, R, D>,
) -> Result<LeftPolar<BoundDynFactor<R, D>>, M::Error>
where
    M: FactorMode<R>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    lease
        .run(|dense| {
            polar_dense(
                engine,
                dense,
                &M::authority(parent.space()),
                parent,
                PolarDirection::Left,
                PolarDirection::Right,
            )
        })
        .map(|(w, p)| LeftPolar { w, p })
}

/// The one dense polar kernel, `(W, P)` per coupled sector from its compact
/// SVD (MatrixAlgebraKit 0.6.8 `implementations/polar.jl` `left_polar!` /
/// `right_polar!(::PolarViaSVD)`): `W = U Vh` and `P = (sqrt(S) Vh)ᴴ (sqrt(S)
/// Vh)` (left) or `(U sqrt(S)) (U sqrt(S))ᴴ` (right), projected Hermitian.
///
/// `authority` admits the direction over every logical coupled sector and
/// builds `P`'s space; the GEMMs write the output regions in place (a packed
/// input whose tree order differs from the output lands by tree extent).
/// One scope spans the per-sector SVDs and products.
fn polar_dense<A, E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    authority: &A,
    input: &BoundDynamicTensorRef<'_, R, D>,
    direction: PolarDirection,
    error_direction: PolarDirection,
) -> Result<DynamicFactorPair<R, D>, A::Error>
where
    A: FactorSpaceAuthority<R>,
    A::Error: From<A::RootError>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let homspace = source_space.homspace();
    // Stored regions omit side-only sectors, whose logical matrices are
    // rows x 0 or 0 x columns and still constrain the isometry direction.
    let rows = authority.coupled_dimensions(homspace.codomain())?;
    let cols = authority.coupled_dimensions(homspace.domain())?;
    for (&sector, &row_count) in &rows {
        if !direction.accepts(row_count, cols.get(&sector).copied().unwrap_or(0)) {
            return Err(error_direction.error().into());
        }
    }
    for (&sector, &col_count) in &cols {
        if !rows.contains_key(&sector) && !direction.accepts(0, col_count) {
            return Err(error_direction.error().into());
        }
    }
    let (p_side, p_leg) = match direction {
        PolarDirection::Left => (FactorSide::Right, homspace.domain()),
        PolarDirection::Right => (FactorSide::Left, homspace.codomain()),
    };
    let p_space = authority.output_space(FusionTreeHomSpace::new(p_leg.clone(), p_leg.clone()))?;
    let matrices =
        generic_input_matricizations(source_space.structure(), input.data(), source_space.nout())?;
    let w_space = authority.same_homspace_output(input.space())?;
    let w_regions = checked_sector_regions(w_space.space().structure(), w_space.space().nout())?
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "polar requires coupled-sector W storage",
        })?;
    let p_nout = p_space.space().nout();
    let p_regions = checked_sector_regions(p_space.space().structure(), p_nout)?.ok_or(
        OperationError::UnsupportedTensorContractScope {
            message: "polar requires coupled-sector P storage",
        },
    )?;
    let w_len = w_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let p_len = p_space
        .space()
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    let (w_landings, p_landings) = with_input_geometry!(&matrices, |geometry| {
        compile_sector_landings(
            geometry,
            &w_regions,
            w_len,
            FactorSide::Left,
            FactorSide::Right,
        )
        .and_then(|w| {
            Ok((
                w,
                compile_sector_landings(geometry, &p_regions, p_len, p_side, p_side)?,
            ))
        })
    })?;
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Polar)?;

    // Why one sector at a time, unlike pinv: polar has no cross-sector
    // cutoff, so only the sector in flight needs its SVD (TensorKit/MAK's
    // working set is the largest block); the outputs are still published
    // only after every sector succeeded.
    let (w_data, p_data) = in_linalg_scope(engine, dense, |dense, engine| {
        // Why zeroed rather than uninitialized: the GEMM destinations are
        // safe initialized views, and every region is then overwritten once.
        let mut w_data = vec![D::zero(); w_len];
        let mut p_data = vec![D::zero(); p_len];
        let mut scratch = Vec::new();
        // Built once: a GEMM configuration owns its axis lists.
        let w_config = DenseDotConfig::matmul();
        let p_config = match direction {
            PolarDirection::Left => DenseDotConfig::matmul().with_conjugation(true, false),
            PolarDirection::Right => DenseDotConfig::matmul().with_conjugation(false, true),
        };
        for (index, (w, p)) in w_landings.iter().zip(&p_landings).enumerate() {
            let matrix = matrices.get(index)?;
            let mut stage =
                compact_svd_numerical_stage(engine, dense, matrix.data, matrix.rows, matrix.cols)?;
            // `W` reads the unscaled factors, so it precedes `P`.
            w.write(&mut w_data, &w_regions[w.output], &mut scratch, |w| {
                polar_isometry(engine, dense, &stage, &w_config, w)
            })?;
            p.write(&mut p_data, &p_regions[p.output], &mut scratch, |p| {
                polar_positive(engine, dense, &mut stage, direction, &p_config, p)
            })?;
        }
        Ok((w_data, p_data))
    })?;

    let w = BoundDynFactor::from_bound(w_space, w_data, source_space.nout(), source_space.nin())?;
    let p = BoundDynFactor::from_bound(p_space, p_data, p_nout, p_nout)?;
    Ok((w, p))
}

pub(super) fn project_hermitian_col_major<D: FactorScalar>(matrix: &mut [D], n: usize) {
    let half = D::from_real(0.5);
    for col in 0..n {
        for row in 0..=col {
            let value =
                (matrix[row + n * col] + FactorScalar::adjoint(matrix[col + n * row])) * half;
            matrix[row + n * col] = value;
            matrix[col + n * row] = FactorScalar::adjoint(value);
        }
    }
}

/// `W = U Vh` into the column-major `rows x cols` destination `w`.
fn polar_isometry<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    stage: &CompactSvdNumericalStage<D>,
    config: &DenseDotConfig,
    w: &mut [D],
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let CompactSvdNumericalStage {
        rows,
        cols,
        rank,
        u,
        vt,
        ..
    } = stage;
    let w_shape = [*rows, *cols];
    let w_strides = [1, *rows];
    let u_shape = [*rows, *rank];
    let u_strides = [1, *rows];
    let vt_shape = [*rank, *cols];
    let vt_strides = [1, *rank];
    let w_view = DenseViewMut::new(w, &w_shape, &w_strides, 0).map_err(OperationError::Dense)?;
    let u_view = DenseView::new(u, &u_shape, &u_strides, 0).map_err(OperationError::Dense)?;
    let vt_view = DenseView::new(vt, &vt_shape, &vt_strides, 0).map_err(OperationError::Dense)?;
    dense
        .dot_general_into(
            engine,
            D::dense_write(w_view),
            D::dense_read(u_view),
            D::dense_read(vt_view),
            config,
        )
        .map_err(OperationError::Dense)
}

/// `P` into the column-major square destination `p`; the `sqrt(S)` scaling
/// is applied in place to the staged factor, which is not read again.
/// `config` is the matmul conjugating the `direction`'s adjoint operand.
fn polar_positive<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    stage: &mut CompactSvdNumericalStage<D>,
    direction: PolarDirection,
    config: &DenseDotConfig,
    p: &mut [D],
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let CompactSvdNumericalStage {
        rows,
        cols,
        rank,
        u,
        singular_values,
        vt,
    } = stage;
    let (rows, cols, rank) = (&*rows, &*cols, &*rank);
    let p_order = match direction {
        PolarDirection::Left => *cols,
        PolarDirection::Right => *rows,
    };
    let x = match direction {
        PolarDirection::Left => vt,
        PolarDirection::Right => u,
    };
    match direction {
        PolarDirection::Left => {
            for (row, sigma) in singular_values.iter().copied().enumerate() {
                let scale = D::from_real(sigma.sqrt());
                for col in 0..*cols {
                    x[row + rank * col] = x[row + rank * col] * scale;
                }
            }
        }
        PolarDirection::Right => {
            for (col, sigma) in singular_values.iter().copied().enumerate() {
                let scale = D::from_real(sigma.sqrt());
                for row in 0..*rows {
                    x[row + rows * col] = x[row + rows * col] * scale;
                }
            }
        }
    }
    let p_shape = [p_order, p_order];
    let p_strides = [1, p_order];
    let p_view = DenseViewMut::new(p, &p_shape, &p_strides, 0).map_err(OperationError::Dense)?;
    match direction {
        PolarDirection::Left => {
            let xh_shape = [*cols, *rank];
            let xh_strides = [*rank, 1];
            let x_shape = [*rank, *cols];
            let x_strides = [1, *rank];
            let xh = DenseView::new(x, &xh_shape, &xh_strides, 0).map_err(OperationError::Dense)?;
            let x = DenseView::new(x, &x_shape, &x_strides, 0).map_err(OperationError::Dense)?;
            dense
                .dot_general_into(
                    engine,
                    D::dense_write(p_view),
                    D::dense_read(xh),
                    D::dense_read(x),
                    config,
                )
                .map_err(OperationError::Dense)?;
        }
        PolarDirection::Right => {
            let x_shape = [*rows, *rank];
            let x_strides = [1, *rows];
            let xh_shape = [*rank, *rows];
            let xh_strides = [*rows, 1];
            let x_view =
                DenseView::new(x, &x_shape, &x_strides, 0).map_err(OperationError::Dense)?;
            let xh = DenseView::new(x, &xh_shape, &xh_strides, 0).map_err(OperationError::Dense)?;
            dense
                .dot_general_into(
                    engine,
                    D::dense_write(p_view),
                    D::dense_read(x_view),
                    D::dense_read(xh),
                    config,
                )
                .map_err(OperationError::Dense)?;
        }
    }
    project_hermitian_col_major(p, p_order);
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) enum PolarDirection {
    Left,
    Right,
}

impl PolarDirection {
    pub(super) fn accepts(self, rows: usize, cols: usize) -> bool {
        match self {
            Self::Left => rows >= cols,
            Self::Right => cols >= rows,
        }
    }

    pub(super) fn error(self) -> OperationError {
        OperationError::InvalidArgument {
            message: match self {
                Self::Left => "left_polar requires rows >= columns in every coupled-sector matrix",
                Self::Right => {
                    "right_polar requires columns >= rows in every coupled-sector matrix"
                }
            },
        }
    }
}
