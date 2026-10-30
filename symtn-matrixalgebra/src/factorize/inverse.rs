use super::*;
use symtn_dense::Engine;

#[derive(Clone, Copy)]
pub(super) struct InverseSectorRoute {
    pub(super) source: usize,
    pub(super) output: usize,
}

#[derive(Clone, Copy)]
pub(super) struct SolveLeftSectorRoute {
    pub(super) divisor: usize,
    pub(super) rhs: usize,
    pub(super) output: usize,
}

pub(super) struct InverseMatrixRoute {
    pub(super) source: usize,
    pub(super) output: usize,
    pub(super) rows: Vec<InverseBasisExtent>,
    pub(super) cols: Vec<InverseBasisExtent>,
}

#[derive(Clone, Copy)]
pub(super) struct InverseBasisExtent {
    pub(super) source_offset: usize,
    pub(super) output_offset: usize,
    pub(super) extent: usize,
}

/// Coefficient-free inverse execution into an already admitted swapped output.
///
/// The caller owns categorical admission; this body only routes the existing
/// coupled-sector layout and performs the dense solves.
pub(crate) fn inverse_by_sector_dyn_into<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let mut output_data = vec![D::zero(); output_space.space().required_len()?];

    let source_regions = checked_sector_regions(source_space.structure(), source_space.nout())?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "inverse derived output requires canonical coupled-sector storage",
    })?;
    match source_regions {
        Some(source) => {
            let routes = compile_inverse_region_routes(
                &source,
                &output_regions,
                input.data().len(),
                output_data.len(),
            )?;
            let max_order = routes
                .iter()
                .map(|route| source[route.source].rows())
                .max()
                .unwrap_or(0);
            let identity = identity_workspace::<D>(max_order)?;
            for route in routes {
                let source_region = &source[route.source];
                if source_region.rows() == 0 {
                    continue;
                }
                let source_matrix = &input.data()[source_region.range()];
                let output_matrix = &mut output_data[output_regions[route.output].range()];
                solve_inverse_sector(
                    engine,
                    dense,
                    source_matrix,
                    output_matrix,
                    source_region.rows(),
                    source_region.rows(),
                    &identity,
                    max_order,
                )?;
            }
        }
        None => {
            let source_matrices =
                sector_matricizations(source_space.structure(), input.data(), source_space.nout())?;
            let routes = compile_inverse_matrix_routes(
                &source_matrices,
                &output_regions,
                output_data.len(),
            )?;
            let max_order = routes
                .iter()
                .map(|route| source_matrices[route.source].rows)
                .max()
                .unwrap_or(0);
            let identity = identity_workspace::<D>(max_order)?;
            let mut solution = vec![D::zero(); identity.len()];
            for route in routes {
                let source = &source_matrices[route.source];
                if source.rows == 0 {
                    continue;
                }
                solve_inverse_sector(
                    engine,
                    dense,
                    &source.data,
                    &mut solution,
                    source.rows,
                    max_order,
                    &identity,
                    max_order,
                )?;
                let output = &output_regions[route.output];
                reorder_inverse_solution(
                    &solution,
                    max_order,
                    &mut output_data[output.range()],
                    output.rows(),
                    &route.rows,
                    &route.cols,
                );
            }
        }
    }

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nin(),
        source_space.nout(),
    )
}

/// `rcond * sigma_max` over every sector's singular values, rejecting a
/// non-finite one in the same pass.
///
/// Why an error rather than NaN propagation: `f64::max` discards NaN, and even
/// a NaN-propagating maximum only yields `sigma > NaN == false`, i.e. a silent
/// all-zero pseudo-inverse. The references are no better — Julia
/// `LinearAlgebra.pinv` (`tol2 = max(rtol*maximum(S), atol)`, `S .> tol2`)
/// zeroes every value under a NaN tolerance — so the only answer that does not
/// hide the NaN is a typed failure, matching `select_truncation`'s
/// `InvalidSpectrum`.
pub(crate) fn pinv_cutoff(
    singular_values: impl IntoIterator<Item = f64>,
    rcond: f64,
) -> Result<f64, OperationError> {
    let sigma_max = singular_values
        .into_iter()
        .try_fold(0.0_f64, |largest, sigma| {
            sigma.is_finite().then(|| largest.max(sigma))
        })
        .ok_or(OperationError::InvalidArgument {
            message: "pinv singular values must be finite",
        })?;
    Ok(rcond * sigma_max)
}

/// The compact pseudo-inverse of an admitted diagonal: every value whose
/// magnitude exceeds the dense route's cutoff ([`pinv_cutoff`], one global
/// `rcond * σ_max`, strict `>`) is inverted by `reciprocal`, the rest are
/// zero. It is the one numeric admission of a compact pseudo-inverse in
/// every fusion mode (#1800):
///
/// - magnitudes are compared unrounded, in `f64`. The dense route compares
///   singular values at the payload's precision, so for a single-precision
///   payload (`f32`, `Complex32`) the two can disagree exactly at the cutoff:
///   a magnitude that rounds up in `f32` onto or past `rcond * σ_max` is cut
///   here and may be kept by the dense route (and conversely). Why the
///   compact arm does not round: rounding would make the cut depend on the
///   payload type rather than the value (#1800);
/// - a retained subnormal stays on this route and inverts to its IEEE
///   reciprocal (`Inf`), as TensorKit `pinv(::DiagonalTensorMap)` and dense
///   `LinearAlgebra.pinv` give;
/// - a nonfinite value is refused by [`pinv_cutoff`]'s typed error, the
///   dense route's.
///
/// TensorKit's own diagonal tolerance rule is not mirrored; see the facade
/// `pinv` documentation for that divergence.
#[doc(hidden)]
pub fn pinv_diagonal_spectrum<D: FactorScalar>(
    spectrum: &[SectorSpectrum<D>],
    rcond: f64,
    reciprocal: impl Fn(D) -> D,
) -> Result<Vec<SectorSpectrum<D>>, OperationError> {
    let magnitude = |value: D| value.widen_complex().norm();
    let cutoff = pinv_cutoff(
        spectrum
            .iter()
            .flat_map(|entry| entry.values.iter().map(|&value| magnitude(value))),
        rcond,
    )?;
    Ok(spectrum
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry
                .values
                .iter()
                .map(|&value| {
                    if magnitude(value) > cutoff {
                        reciprocal(value)
                    } else {
                        D::zero()
                    }
                })
                .collect(),
        })
        .collect())
}

/// Coefficient-free pseudo-inverse into an already admitted swapped space.
///
/// All layout checks happen before staging or dense work.  The local staging
/// keeps an SVD or GEMM failure from publishing a partial output.
pub(crate) fn pinv_by_sector_dyn_into<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    pinv_oriented_by_sector_dyn_into(
        engine,
        dense,
        input,
        output_space,
        rcond,
        FactorPlacement::Direct,
    )
}

/// The pseudo-inverse of the logical adjoint of `parent`, read in place:
/// `A = U S Vh` gives `(A^H)^+ = U S^+ Vh`, in `parent`'s own hom space.
pub(crate) fn pinv_adjoint_by_sector_dyn_into<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    parent: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    rcond: f64,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    pinv_oriented_by_sector_dyn_into(
        engine,
        dense,
        parent,
        output_space,
        rcond,
        FactorPlacement::Adjoint,
    )
}

fn pinv_oriented_by_sector_dyn_into<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    rcond: f64,
    placement: FactorPlacement,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let source_space = input.space().space();
    let Some(source_identity) = source_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(output_identity) = output_space.space().admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    if source_identity != output_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: source_identity.clone(),
                actual: output_identity.clone(),
            },
        ));
    }
    if !Arc::ptr_eq(input.space().provider_arc(), output_space.provider_arc()) {
        return Err(OperationError::StructureMismatch {
            tensor: "pinv output provider authority",
        });
    }
    let expected_homspace = match placement {
        FactorPlacement::Direct => FusionTreeHomSpace::new(
            source_space.homspace().domain().clone(),
            source_space.homspace().codomain().clone(),
        ),
        FactorPlacement::Adjoint => source_space.homspace().clone(),
    };
    let expected_nout = expected_homspace.codomain().len();
    let expected_nin = expected_homspace.domain().len();
    if output_space.space().nout() != expected_nout
        || output_space.space().nin() != expected_nin
        || output_space.space().rank() != expected_nout + expected_nin
        || output_space.space().homspace() != &expected_homspace
    {
        return Err(OperationError::StructureMismatch {
            tensor: "pinv output space",
        });
    }
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "pinv requires coupled-sector output storage",
    })?;
    let output_len = output_space.space().required_len()?;
    // The one finite-input stage (#1986), before the provider: a provider's SVD
    // may refuse a nonfinite input or return NaN singular values, so only this
    // check gives every provider the compact route's kind (#1800).
    require_finite_factor_input(input.data().iter().copied(), FactorFamily::Svd).map_err(|_| {
        OperationError::InvalidArgument {
            message: "pinv singular values must be finite",
        }
    })?;
    let matrices =
        generic_input_matricizations(source_space.structure(), input.data(), source_space.nout())?;
    // `A^+` is `domain <- codomain`: its rows are the input's column trees.
    let (rows, cols) = match placement {
        FactorPlacement::Direct => (FactorSide::Right, FactorSide::Left),
        FactorPlacement::Adjoint => (FactorSide::Left, FactorSide::Right),
    };
    let landings = with_input_geometry!(&matrices, |geometry| compile_sector_landings(
        geometry,
        &output_regions,
        output_len,
        rows,
        cols,
    ))?;

    let mut output_data = vec![D::zero(); output_len];
    let mut scratch = Vec::new();
    // One scope spans both passes: the staged SVDs (the cutoff is global) and
    // the per-sector reconstruction GEMMs.
    in_linalg_scope(engine, dense, |dense, engine| {
        let mut staged = Vec::with_capacity(matrices.len());
        for index in 0..matrices.len() {
            let matrix = matrices.get(index)?;
            let (rows, cols) = (matrix.rows, matrix.cols);
            let rank = rows.min(cols);
            // Why no SVD gauge: the pseudo-inverse is gauge invariant.
            let (u, singular_values, vt) = if rank == 0 {
                (Vec::new(), Vec::new(), Vec::new())
            } else {
                compact_svd_owned(engine, dense, matrix.data, rows, cols)?
            };
            staged.push(CompactSvdNumericalStage {
                rows,
                cols,
                rank,
                u,
                singular_values,
                vt,
            });
        }
        let cutoff = pinv_cutoff(
            staged
                .iter()
                .flat_map(|stage| stage.singular_values.iter().copied()),
            rcond,
        )?;
        for (stage, landing) in staged.iter_mut().zip(&landings) {
            if stage.rank == 0 {
                continue;
            }
            let CompactSvdNumericalStage {
                rows,
                cols,
                rank,
                u,
                singular_values,
                vt,
            } = stage;
            let (rows, cols, rank) = (*rows, *cols, *rank);
            for (row, &sigma) in singular_values.iter().enumerate() {
                let reciprocal = D::from_real(if sigma > cutoff { 1.0 / sigma } else { 0.0 });
                for column in 0..cols {
                    vt[row + rank * column] = vt[row + rank * column] * reciprocal;
                }
            }
            // `A^+ = V S^+ U^H` reads `S^+ Vh` and `U` conjugated and
            // transposed; `(A^H)^+ = U S^+ Vh` reads them as stored.
            let (out_rows, out_cols, left, left_strides, right, right_strides, conjugate) =
                match placement {
                    FactorPlacement::Direct => (cols, rows, &*vt, [rank, 1], &*u, [rows, 1], true),
                    FactorPlacement::Adjoint => {
                        (rows, cols, &*u, [1, rows], &*vt, [1, rank], false)
                    }
                };
            landing.write(
                &mut output_data,
                &output_regions[landing.output],
                &mut scratch,
                |output| {
                    let output_shape = [out_rows, out_cols];
                    let output_strides = [1, out_rows];
                    let left_shape = [out_rows, rank];
                    let right_shape = [rank, out_cols];
                    let output_view = DenseViewMut::new(output, &output_shape, &output_strides, 0)
                        .map_err(OperationError::Dense)?;
                    let left_view = DenseView::new(left, &left_shape, &left_strides, 0)
                        .map_err(OperationError::Dense)?;
                    let right_view = DenseView::new(right, &right_shape, &right_strides, 0)
                        .map_err(OperationError::Dense)?;
                    dense
                        .dot_general_into(
                            engine,
                            D::dense_write(output_view),
                            D::dense_read(left_view),
                            D::dense_read(right_view),
                            &DenseDotConfig::matmul().with_conjugation(conjugate, conjugate),
                        )
                        .map_err(OperationError::Dense)
                },
            )?;
        }
        Ok(())
    })?;
    let (nout, nin) = match placement {
        FactorPlacement::Direct => (source_space.nin(), source_space.nout()),
        FactorPlacement::Adjoint => (source_space.nout(), source_space.nin()),
    };
    BoundDynFactor::from_bound(output_space, output_data, nout, nin)
}

pub(crate) fn solve_left_by_sector_dyn_into<E, R, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    divisor: &BoundDynamicTensorRef<'_, R, D>,
    rhs: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let divisor_space = divisor.space().space();
    let rhs_space = rhs.space().space();
    let Some(divisor_identity) = divisor_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(rhs_identity) = rhs_space.admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    let Some(output_identity) = output_space.space().admission().rule_identity() else {
        return Err(OperationError::from_core_preserving_context(
            CoreError::MissingFusionRuleIdentity,
        ));
    };
    if divisor_identity != rhs_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: divisor_identity.clone(),
                actual: rhs_identity.clone(),
            },
        ));
    }
    if divisor_identity != output_identity {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionRuleMismatch {
                expected: divisor_identity.clone(),
                actual: output_identity.clone(),
            },
        ));
    }
    if !Arc::ptr_eq(divisor.space().provider_arc(), output_space.provider_arc()) {
        return Err(OperationError::StructureMismatch {
            tensor: "solve output provider authority",
        });
    }
    if divisor_space.homspace().codomain() != rhs_space.homspace().codomain() {
        return Err(OperationError::SpaceMismatch {
            message: "solve requires equal divisor and right-hand-side codomains",
        });
    }
    let expected_homspace = FusionTreeHomSpace::new(
        divisor_space.homspace().domain().clone(),
        rhs_space.homspace().domain().clone(),
    );
    let expected_nout = expected_homspace.codomain().len();
    let expected_nin = expected_homspace.domain().len();
    if output_space.space().nout() != expected_nout
        || output_space.space().nin() != expected_nin
        || output_space.space().rank() != expected_nout + expected_nin
        || output_space.space().homspace() != &expected_homspace
    {
        return Err(OperationError::StructureMismatch {
            tensor: "solve output space",
        });
    }

    let divisor_regions = checked_sector_regions(divisor_space.structure(), divisor_space.nout())?
        .ok_or(OperationError::UnsupportedTensorContractScope {
            message: "solve requires canonical coupled-sector divisor storage",
        })?;
    for region in divisor_regions.iter() {
        validate_region_range(region, divisor.data().len())?;
        if region.rows() != region.cols() {
            return Err(OperationError::SpaceMismatch {
                message: "solve coupled-sector divisor matrices must be square",
            });
        }
    }
    let rhs_regions = checked_sector_regions(rhs_space.structure(), rhs_space.nout())?.ok_or(
        OperationError::UnsupportedTensorContractScope {
            message: "solve requires canonical coupled-sector right-hand-side storage",
        },
    )?;
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "solve derived output requires canonical coupled-sector storage",
    })?;
    let output_len = output_space.space().required_len()?;
    let routes = compile_solve_left_region_routes(
        &divisor_regions,
        &rhs_regions,
        &output_regions,
        divisor.data().len(),
        rhs.data().len(),
        output_len,
    )?;

    let mut output_data = vec![D::zero(); output_len];

    for route in routes {
        let divisor_region = &divisor_regions[route.divisor];
        if divisor_region.rows() == 0 || output_regions[route.output].cols() == 0 {
            continue;
        }
        solve_left_sector(
            engine,
            dense,
            &divisor.data()[divisor_region.range()],
            &rhs.data()[rhs_regions[route.rhs].range()],
            &mut output_data[output_regions[route.output].range()],
            divisor_region.rows(),
            rhs_regions[route.rhs].cols(),
        )?;
    }

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        divisor_space.nin(),
        rhs_space.nin(),
    )
}

pub(super) fn compile_solve_left_region_routes(
    divisor: &[CoupledSectorRegion],
    rhs: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    divisor_len: usize,
    rhs_len: usize,
    output_len: usize,
) -> Result<Vec<SolveLeftSectorRoute>, OperationError> {
    let divisor_by_sector = sector_region_index_map(divisor)?;
    let rhs_by_sector = sector_region_index_map(rhs)?;
    let mut routes = Vec::with_capacity(output.len());
    for (output_index, output_region) in output.iter().enumerate() {
        let sector = output_region.coupled();
        let divisor_index =
            divisor_by_sector
                .get(&sector)
                .copied()
                .ok_or(OperationError::SpaceMismatch {
                    message: "solve divisor is missing an output coupled sector",
                })?;
        let rhs_index =
            rhs_by_sector
                .get(&sector)
                .copied()
                .ok_or(OperationError::SpaceMismatch {
                    message: "solve right-hand side is missing an output coupled sector",
                })?;
        let divisor_region = &divisor[divisor_index];
        let rhs_region = &rhs[rhs_index];
        if divisor_region.rows() != divisor_region.cols()
            || rhs_region.rows() != divisor_region.rows()
            || output_region.rows() != divisor_region.cols()
            || output_region.cols() != rhs_region.cols()
        {
            return Err(OperationError::SpaceMismatch {
                message: "solve coupled-sector matrix dimensions are incompatible",
            });
        }
        if divisor_region.row_trees() != rhs_region.row_trees()
            || divisor_region.col_trees() != output_region.row_trees()
            || rhs_region.col_trees() != output_region.col_trees()
        {
            return Err(OperationError::SpaceMismatch {
                message: "solve coupled-sector tree bases are incompatible",
            });
        }
        validate_region_range(divisor_region, divisor_len)?;
        validate_region_range(rhs_region, rhs_len)?;
        validate_region_range(output_region, output_len)?;
        routes.push(SolveLeftSectorRoute {
            divisor: divisor_index,
            rhs: rhs_index,
            output: output_index,
        });
    }
    if rhs_by_sector.len() != routes.len() {
        return Err(OperationError::SpaceMismatch {
            message: "solve right-hand side contains a sector absent from the output",
        });
    }
    Ok(routes)
}

pub(super) fn solve_left_sector<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    divisor: &[D],
    rhs: &[D],
    output: &mut [D],
    order: usize,
    columns: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let divisor_shape = [order, order];
    let rhs_shape = [order, columns];
    let divisor_strides = [1, order];
    let rhs_strides = [1, order];
    let divisor = DenseView::new(divisor, &divisor_shape, &divisor_strides, 0)
        .map_err(OperationError::Dense)?;
    let rhs = DenseView::new(rhs, &rhs_shape, &rhs_strides, 0).map_err(OperationError::Dense)?;
    let output =
        DenseViewMut::new(output, &rhs_shape, &rhs_strides, 0).map_err(OperationError::Dense)?;
    dense
        .solve_into(
            engine,
            D::dense_read(divisor),
            D::dense_read(rhs),
            D::dense_write(output),
        )
        .map_err(OperationError::Dense)
}

/// Walks the coupled-sector matricization of an endomorphism and replaces each
/// square block by `apply`'s image of it, writing into `output_space`.
///
/// This is the block-data half of a per-sector matrix function — the caller
/// supplies only the dense `n x n -> n x n` kernel and never sees the layout.
/// `init` is handed the largest block order once, before the loop, so a
/// kernel can size its scratch to `O(max_c n_c²)` and allocate nothing per
/// sector.
///
/// That bound is the kernel's own. It is also the whole of the scratch on a
/// canonical region layout whose trees the output lists in the same order:
/// the blocks are read and written in place. A packed input matricizes
/// *every* sector up front and so costs `O(Σ_c n_c²)` on top of it, and a
/// sector whose tree order differs from the output's lands through one
/// `O(max_c n_c²)` staging buffer.
///
/// `apply(dense, state, source, n, out)` reads the column-major `n x n` block at
/// `source` and writes its image into the column-major `n x n` `out`; one
/// admitted dense scope spans every sector.
///
/// Publication is atomic: the result tensor is built only after every sector
/// has succeeded, so a failure in the last block leaves no half-written tensor
/// behind and never mutates the input.
///
/// Refusing a non-endomorphism is the caller's job, so that the message
/// names the function the user called (`exp`) rather than whichever helper
/// noticed first.
pub(crate) fn map_square_sectors_dyn_into<E, R, D, S, I, F>(
    engine: &mut dyn Engine,
    dense: &mut E,
    input: &BoundDynamicTensorRef<'_, R, D>,
    output_space: BoundDynamicFusionMapSpace<R>,
    init: I,
    mut apply: F,
) -> Result<BoundDynFactor<R, D>, OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
    S: Send,
    I: FnOnce(usize) -> Result<S, OperationError>,
    F: FnMut(
            &mut dyn DenseExecutor,
            &mut dyn Engine,
            &mut S,
            &[D],
            usize,
            &mut [D],
        ) -> Result<(), OperationError>
        + Send,
{
    let source_space = input.space().space();
    if source_space.homspace() != output_space.space().homspace() {
        return Err(OperationError::SpaceMismatch {
            message: "matrix function output must preserve the input homspace",
        });
    }

    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "matrix function output requires canonical coupled-sector storage",
    })?;
    let output_len = output_space.space().required_len()?;
    let matrices =
        generic_input_matricizations(source_space.structure(), input.data(), source_space.nout())?;
    // Why stacking and not only tree identity, as `inv` needs: `f(P_R A
    // P_C^T)` is not `P_R f(A) P_C^T` unless the row and column stackings
    // coincide.
    matrices.validate_endomorphism_stacking(EndomorphismFamily::Exp)?;
    let landings = with_input_geometry!(&matrices, |geometry| compile_sector_landings(
        geometry,
        &output_regions,
        output_len,
        FactorSide::Left,
        FactorSide::Right,
    ))?;
    let max_order = (0..matrices.len())
        .map(|index| matrices.get(index).map(|matrix| matrix.rows))
        .try_fold(0, |largest, rows| rows.map(|rows| largest.max(rows)))?;
    let mut state = init(max_order)?;
    // One admitted dense scope spans every sector's kernel.
    let output_data = in_linalg_scope(engine, dense, |dense, engine| {
        let mut output_data = vec![D::zero(); output_len];
        let mut scratch = Vec::new();
        for (index, landing) in landings.iter().enumerate() {
            let matrix = matrices.get(index)?;
            let order = matrix.rows;
            if order == 0 {
                continue;
            }
            landing.write(
                &mut output_data,
                &output_regions[landing.output],
                &mut scratch,
                |output| apply(dense, engine, &mut state, matrix.data, order, output),
            )?;
        }
        Ok(output_data)
    })?;

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nout(),
        source_space.nin(),
    )
}

pub(super) fn compile_inverse_region_routes(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
    source_len: usize,
    output_len: usize,
) -> Result<Vec<InverseSectorRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_region) in source.iter().enumerate() {
        let output_index = output_by_sector
            .get(&source_region.coupled())
            .copied()
            .ok_or(OperationError::SpaceMismatch {
                message: "inverse output is missing a source coupled sector",
            })?;
        validate_inverse_region(source_region, &output[output_index])?;
        validate_region_range(source_region, source_len)?;
        validate_region_range(&output[output_index], output_len)?;
        used[output_index] = true;
        routes.push(InverseSectorRoute {
            source: source_index,
            output: output_index,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::SpaceMismatch {
            message: "inverse output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

#[cfg(test)]
pub(crate) fn validate_inverse_region_routes_for_test(
    source: &[CoupledSectorRegion],
    output: &[CoupledSectorRegion],
) -> Result<(), OperationError> {
    let source_len = source
        .iter()
        .map(|region| region.range().end)
        .max()
        .unwrap_or(0);
    let output_len = output
        .iter()
        .map(|region| region.range().end)
        .max()
        .unwrap_or(0);
    compile_inverse_region_routes(source, output, source_len, output_len).map(|_| ())
}

pub(super) fn validate_inverse_region(
    source: &CoupledSectorRegion,
    output: &CoupledSectorRegion,
) -> Result<(), OperationError> {
    if source.rows() != source.cols()
        || output.rows() != source.cols()
        || output.cols() != source.rows()
    {
        return Err(OperationError::SpaceMismatch {
            message: "inverse coupled-sector matrix is not square",
        });
    }
    if source.col_trees() != output.row_trees() || source.row_trees() != output.col_trees() {
        return Err(OperationError::SpaceMismatch {
            message: "inverse output tree basis does not transpose the source basis",
        });
    }
    Ok(())
}

pub(super) fn validate_region_range(
    region: &CoupledSectorRegion,
    data_len: usize,
) -> Result<(), OperationError> {
    let range = region.range();
    if range.end > data_len {
        return Err(OperationError::ElementCountMismatch {
            expected: range.end,
            actual: data_len,
        });
    }
    Ok(())
}

pub(super) fn compile_inverse_matrix_routes<D>(
    source: &[SectorMatricization<D>],
    output: &[CoupledSectorRegion],
    output_len: usize,
) -> Result<Vec<InverseMatrixRoute>, OperationError> {
    let output_by_sector = sector_region_index_map(output)?;
    let mut used = vec![false; output.len()];
    let mut routes = Vec::with_capacity(source.len());
    for (source_index, source_matrix) in source.iter().enumerate() {
        let output_index = output_by_sector.get(&source_matrix.sector).copied().ok_or(
            OperationError::SpaceMismatch {
                message: "inverse output is missing a source coupled sector",
            },
        )?;
        let output_region = &output[output_index];
        if source_matrix.rows != source_matrix.cols
            || output_region.rows() != source_matrix.cols
            || output_region.cols() != source_matrix.rows
        {
            return Err(OperationError::SpaceMismatch {
                message: "inverse coupled-sector matrix is not square",
            });
        }
        validate_region_range(output_region, output_len)?;
        let rows =
            compile_basis_extents(source_matrix, FactorSide::Right, output_region.row_trees())?;
        let cols =
            compile_basis_extents(source_matrix, FactorSide::Left, output_region.col_trees())?;
        used[output_index] = true;
        routes.push(InverseMatrixRoute {
            source: source_index,
            output: output_index,
            rows,
            cols,
        });
    }
    if used.iter().any(|used| !used) {
        return Err(OperationError::SpaceMismatch {
            message: "inverse output contains a coupled sector absent from the source",
        });
    }
    Ok(routes)
}

pub(super) fn identity_workspace<D: FactorScalar>(order: usize) -> Result<Vec<D>, OperationError> {
    let elements = order
        .checked_mul(order)
        .ok_or(OperationError::ElementCountOverflow)?;
    let mut identity = vec![D::zero(); elements];
    for index in 0..order {
        identity[index + order * index] = D::from_real(1.0);
    }
    Ok(identity)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn solve_inverse_sector<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    source: &[D],
    output: &mut [D],
    order: usize,
    output_leading: usize,
    identity: &[D],
    identity_order: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
{
    let shape = [order, order];
    let matrix_strides = [1, order];
    let output_strides = [1, output_leading];
    let identity_strides = [1, identity_order];
    let source =
        DenseView::new(source, &shape, &matrix_strides, 0).map_err(OperationError::Dense)?;
    let identity =
        DenseView::new(identity, &shape, &identity_strides, 0).map_err(OperationError::Dense)?;
    let output =
        DenseViewMut::new(output, &shape, &output_strides, 0).map_err(OperationError::Dense)?;
    dense
        .solve_into(
            engine,
            D::dense_read(source),
            D::dense_read(identity),
            D::dense_write(output),
        )
        .map_err(OperationError::Dense)
}

pub(super) fn reorder_inverse_solution<D: Copy>(
    source: &[D],
    source_rows: usize,
    output: &mut [D],
    output_rows: usize,
    row_extents: &[InverseBasisExtent],
    col_extents: &[InverseBasisExtent],
) {
    for rows in row_extents {
        for cols in col_extents {
            for col in 0..cols.extent {
                let source_start = rows.source_offset + source_rows * (cols.source_offset + col);
                let output_start = rows.output_offset + output_rows * (cols.output_offset + col);
                output[output_start..output_start + rows.extent]
                    .copy_from_slice(&source[source_start..source_start + rows.extent]);
            }
        }
    }
}
