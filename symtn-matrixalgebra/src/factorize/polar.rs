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

/// Polar factors of `source` in fusion mode `M` in `direction`, as `(W, P)`
/// (`(Wh, P)` for the right polar). A compact diagonal's phases and
/// magnitudes stay compact on the input space. A lazy adjoint `A^H` is the
/// parent's opposite polar, `left_polar(A^H) = (wh^H, p)` of
/// `right_polar(A) = (p, wh)`: the parent's isometry is returned as
/// [`FactorOutput::DenseAdjoint`], and direction errors name the requested
/// operation, not the parent's.
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
    // Every dense route checks values after its direction admission.
    factor_from_source(
        engine,
        lease,
        source,
        None,
        |space, spectrum| {
            let LeftPolar { w, p } = left_polar_of_diagonal::<M, R, D>(space, spectrum)?;
            Ok((w, p))
        },
        |engine, dense, input| {
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
        Some(
            |engine: &mut dyn Engine,
             dense: &mut E,
             parent: &BoundDynamicTensorRef<'_, R, D>,
             _: &BoundDynamicFusionMapSpace<R>| {
                polar_dense(
                    engine,
                    dense,
                    &M::authority(parent.space()),
                    parent,
                    direction.opposite(),
                    direction,
                )
                .map(|(w, p)| (FactorOutput::DenseAdjoint(w), FactorOutput::Dense(p)))
            },
        ),
    )
}

impl<R, D: FactorScalar> FactorOp<R, D> for op::LeftPolar {
    type Output = LeftPolar<FactorOutput<R, D>>;
}

/// Left polar decomposition `W * P` in fusion mode `M` (MatrixAlgebraKit
/// `left_polar!` via `PolarViaSVD`).
impl<R, D: FactorScalar> Run<R, D> for op::LeftPolar {
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
        polar_from_source::<M, _, _, _, _>(engine, lease, source, PolarDirection::Left)
            .map(|(w, p)| LeftPolar { w, p })
    }
}

impl<R, D: FactorScalar> FactorOp<R, D> for op::RightPolar {
    type Output = RightPolar<FactorOutput<R, D>>;
}

/// Right polar decomposition `P * Wh` in fusion mode `M` (MatrixAlgebraKit
/// `right_polar!` via `PolarViaSVD`).
impl<R, D: FactorScalar> Run<R, D> for op::RightPolar {
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
        polar_from_source::<M, _, _, _, _>(engine, lease, source, PolarDirection::Right)
            .map(|(wh, p)| RightPolar { p, wh })
    }
}

/// The one dense polar kernel, `(W, P)` per coupled sector from its compact
/// SVD (MatrixAlgebraKit 0.6.9 `implementations/polar.jl` `left_polar!` /
/// `right_polar!(::PolarViaSVD)`): `W = U Vh` and `P = (sqrt(S) Vh)ᴴ (sqrt(S)
/// Vh)` (left) or `(U sqrt(S)) (U sqrt(S))ᴴ` (right), projected Hermitian.
///
/// `authority` admits the direction over every logical coupled sector and
/// builds `P`'s space. Every sector of nonzero rank is one input of a
/// [`DenseFactorization::LeftPolar`] / [`DenseFactorization::RightPolar`]
/// destination batch writing the output regions in place. One scope spans the
/// batch.
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

    // Every sector with a nonzero rank is one input of one destination
    // batch; a sector of rank zero has empty factors and is not solved.
    let (_, blocks) = sector_blocks(&matrices)?;
    let solved: Vec<usize> = (0..blocks.len())
        .filter(|&index| blocks[index].1.min(blocks[index].2) > 0)
        .collect();
    let solved_blocks: Vec<_> = solved.iter().map(|&index| blocks[index]).collect();
    let mut layouts = Vec::new();
    let inputs = col_major_reads(&solved_blocks, &mut layouts)?;
    let op = match direction {
        PolarDirection::Left => DenseFactorization::LeftPolar,
        PolarDirection::Right => DenseFactorization::RightPolar,
    };
    // Why allocator zeroing rather than uninitialized outputs: the
    // destinations are safe initialized views, and the zero pages of a large
    // output are first touched by the lane that writes them.
    let mut w_data = zeroed_payload::<D>(w_len);
    let mut p_data = zeroed_payload::<D>(p_len);
    {
        let range_of = |landings: &[SectorLanding], regions: &[CoupledSectorRegion]| {
            landings
                .iter()
                .map(|landing| regions[landing.output].range())
                .collect::<Vec<_>>()
        };
        let mut w_slices = split_ranges(&mut w_data, &range_of(&w_landings, &w_regions))?;
        let mut p_slices = split_ranges(&mut p_data, &range_of(&p_landings, &p_regions))?;
        let dest_layouts = solved_blocks
            .iter()
            .map(|&(_, rows, cols)| {
                let (w, p) = match direction {
                    PolarDirection::Left => (matrix_layout(rows, cols), matrix_layout(cols, cols)),
                    PolarDirection::Right => (matrix_layout(rows, cols), matrix_layout(rows, rows)),
                };
                [w, p]
            })
            .collect::<Vec<_>>();
        let mut dests = solved
            .iter()
            .zip(&dest_layouts)
            .map(|(&index, [w_layout, p_layout])| {
                let w = std::mem::take(&mut w_slices[index]);
                let p = std::mem::take(&mut p_slices[index]);
                let (w, p) = (dest_write(w, w_layout)?, dest_write(p, p_layout)?);
                let outputs = match direction {
                    PolarDirection::Left => [Some(w), Some(p), None],
                    PolarDirection::Right => [Some(p), Some(w), None],
                };
                Ok(FactorDest::new(
                    outputs,
                    symtn_dense::FactorPlacement::AsComputed,
                ))
            })
            .collect::<Result<Vec<_>, OperationError>>()?;
        // Why one scope around the batch: the backend admits its dense work
        // once, and the scope's own refusal takes precedence over the
        // batch's error.
        in_linalg_scope(engine, dense, |dense, engine| {
            factorize_batch_into_staged(engine, dense, op, &inputs, &mut dests, |_, _, outcome| {
                // The first failure of any kind, in sector order, is the
                // operation's error.
                outcome.map_err(|failure| OperationError::Dense(failure.into_error()))?;
                Ok(())
            })
        })?;
    }

    let w = BoundDynFactor::from_bound(w_space, w_data, source_space.nout(), source_space.nin())?;
    let p = BoundDynFactor::from_bound(p_space, p_data, p_nout, p_nout)?;
    Ok((w, p))
}

#[derive(Clone, Copy)]
pub(super) enum PolarDirection {
    Left,
    Right,
}

impl PolarDirection {
    /// The direction of the parent's polar that yields this one of its
    /// adjoint.
    fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }

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
