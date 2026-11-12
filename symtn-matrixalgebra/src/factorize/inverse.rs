use super::*;
use symtn_dense::Engine;

#[derive(Clone, Copy)]
pub(super) struct SolveLeftSectorRoute {
    pub(super) divisor: usize,
    pub(super) rhs: usize,
    pub(super) output: usize,
}

/// The inverse (TensorKit `inv`) of a map between isomorphic spaces, every
/// coupled sector of nonzero order one input of a [`DenseFactorization::Inv`]
/// destination batch writing its output region in place. One scope spans the
/// batch, and the first solver failure in sector order is the error.
///
/// The caller owns categorical admission; this body only routes the existing
/// coupled-sector layout: the output's rows are the input's column trees and
/// its columns the input's row trees, at the same offsets.
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
    let output_regions = checked_sector_regions(
        output_space.space().structure(),
        output_space.space().nout(),
    )?
    .ok_or(OperationError::UnsupportedTensorContractScope {
        message: "inverse derived output requires canonical coupled-sector storage",
    })?;
    let output_len = output_space.space().required_len()?;
    let matrices =
        generic_input_matricizations(source_space.structure(), input.data(), source_space.nout())?;
    let (_, blocks) = sector_blocks(&matrices)?;
    if blocks.iter().any(|&(_, rows, cols)| rows != cols) {
        return Err(OperationError::SpaceMismatch {
            message: "inverse coupled-sector matrix is not square",
        });
    }
    let landings = with_input_geometry!(&matrices, |geometry| compile_sector_landings(
        geometry,
        &output_regions,
        output_len,
        FactorSide::Right,
        FactorSide::Left,
    ))?;
    let solved: Vec<usize> = (0..blocks.len())
        .filter(|&index| blocks[index].1 > 0)
        .collect();
    let solved_blocks: Vec<_> = solved.iter().map(|&index| blocks[index]).collect();
    let mut layouts = Vec::new();
    let inputs = col_major_reads(&solved_blocks, &mut layouts)?;
    // Why allocator zeroing rather than uninitialized outputs: see
    // `exp_via_eigh_dyn_into`.
    let mut output_data = zeroed_payload::<D>(output_len);
    if !inputs.is_empty() {
        let ranges: Vec<_> = landings
            .iter()
            .map(|landing| output_regions[landing.output].range())
            .collect();
        let mut slices = split_ranges(&mut output_data, &ranges)?;
        let dest_layouts: Vec<_> = solved_blocks
            .iter()
            .map(|&(_, order, _)| matrix_layout(order, order))
            .collect();
        let mut dests = solved
            .iter()
            .zip(&dest_layouts)
            .map(|(&index, layout)| {
                Ok(FactorDest::new(
                    [
                        Some(dest_write(std::mem::take(&mut slices[index]), layout)?),
                        None,
                        None,
                    ],
                    symtn_dense::FactorPlacement::AsComputed,
                ))
            })
            .collect::<Result<Vec<_>, OperationError>>()?;
        in_linalg_scope(engine, dense, |dense, engine| {
            factorize_into_solver_first(
                engine,
                dense,
                DenseFactorization::Inv,
                &inputs,
                &mut dests,
                |_, _| Ok(()),
            )
        })?;
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
pub(crate) fn pinv_diagonal_spectrum<D: FactorScalar>(
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
    // One scope spans both passes: the staged SVDs (the cutoff is global) and
    // the per-sector reconstruction GEMMs.
    in_linalg_scope(engine, dense, |dense, engine| {
        let (_, blocks) = sector_blocks(&matrices)?;
        // Each sector's geometry and spectrum; its factors stay in the
        // owned batch's per-sector outputs until the global cutoff is known,
        // and the reconstruction GEMMs read them in place.
        let mut staged = Vec::with_capacity(blocks.len());
        // Why no SVD gauge: the pseudo-inverse is gauge invariant.
        let mut factors = compact_svd_sectors_staged(
            engine,
            dense,
            &blocks,
            None,
            SvdStorage::PerSector,
            |_, _| {},
            |_, stage| {
                staged.push((
                    stage.rows,
                    stage.cols,
                    stage.rank,
                    stage.u_range,
                    stage.singular_values,
                    stage.vt_range,
                ));
                Ok(())
            },
        )?;
        let cutoff = pinv_cutoff(
            staged.iter().flat_map(|stage| stage.4.iter().copied()),
            rcond,
        )?;
        for (position, (stage, landing)) in staged.iter().zip(&landings).enumerate() {
            let (rows, cols, rank, ref u_range, ref singular_values, ref vt_range) = *stage;
            if rank == 0 {
                continue;
            }
            let (u, vt) = factors.sector_mut(position, u_range.clone(), vt_range.clone());
            let u = &*u;
            for (row, &sigma) in singular_values.iter().enumerate() {
                let reciprocal = D::from_real(if sigma > cutoff { 1.0 / sigma } else { 0.0 });
                for column in 0..cols {
                    vt[row + rank * column] = vt[row + rank * column] * reciprocal;
                }
            }
            let vt = &*vt;
            // `A^+ = V S^+ U^H` reads `S^+ Vh` and `U` conjugated and
            // transposed; `(A^H)^+ = U S^+ Vh` reads them as stored.
            let (out_rows, out_cols, left, left_strides, right, right_strides, conjugate) =
                match placement {
                    FactorPlacement::Direct => (cols, rows, vt, [rank, 1], u, [rows, 1], true),
                    FactorPlacement::Adjoint => (rows, cols, u, [1, rows], vt, [1, rank], false),
                };
            let output = &mut output_data[output_regions[landing.output].range()];
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
                .map_err(OperationError::Dense)?;
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

    // Every nonempty system writes its whole region and an empty region has
    // no elements, so allocator zeroing replaces a serial fill.
    let mut output_data = zeroed_payload::<D>(output_len);
    let solved: Vec<_> = routes
        .iter()
        .filter(|route| {
            divisor_regions[route.divisor].rows() > 0 && output_regions[route.output].cols() > 0
        })
        .collect();
    if !solved.is_empty() {
        // `[n, n]`, `[n, k]` and the column-major strides `[1, n]` of each.
        let layouts: Vec<_> = solved
            .iter()
            .map(|route| {
                let n = divisor_regions[route.divisor].rows();
                ([n, n], [n, rhs_regions[route.rhs].cols()], [1, n])
            })
            .collect();
        fn reads<'a, D: FactorScalar>(
            data: &'a [D],
            range: std::ops::Range<usize>,
            shape: &'a [usize],
            strides: &'a [usize],
        ) -> Result<DenseRead<'a>, OperationError> {
            DenseView::new(&data[range], shape, strides, 0)
                .map(D::dense_read)
                .map_err(OperationError::Dense)
        }
        let divisors = solved
            .iter()
            .zip(&layouts)
            .map(|(route, (square, _, strides))| {
                reads(
                    divisor.data(),
                    divisor_regions[route.divisor].range(),
                    square,
                    strides,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rhs_reads = solved
            .iter()
            .zip(&layouts)
            .map(|(route, (_, shape, strides))| {
                reads(rhs.data(), rhs_regions[route.rhs].range(), shape, strides)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ranges: Vec<_> = solved
            .iter()
            .map(|route| output_regions[route.output].range())
            .collect();
        let mut outputs = split_ranges(&mut output_data, &ranges)?
            .into_iter()
            .zip(&layouts)
            .map(|(slice, (_, shape, strides))| {
                DenseViewMut::new(slice, shape, strides, 0)
                    .map(D::dense_write)
                    .map_err(OperationError::Dense)
            })
            .collect::<Result<Vec<_>, _>>()?;
        dense
            .solve_batch_into(engine, &divisors, &rhs_reads, &mut outputs)
            .map_err(OperationError::Dense)?;
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
/// canonical region layout: the blocks are read and written in place. A
/// packed input matricizes *every* sector up front and so costs
/// `O(Σ_c n_c²)` on top of it.
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
    let SquareSectorRoutes {
        output_regions,
        output_len,
        matrices,
        landings,
    } = square_sector_routes(input, &output_space)?;
    let max_order = (0..matrices.len())
        .map(|index| matrices.get(index).map(|matrix| matrix.rows))
        .try_fold(0, |largest, rows| rows.map(|rows| largest.max(rows)))?;
    let mut state = init(max_order)?;
    // One admitted dense scope spans every sector's kernel.
    let output_data = in_linalg_scope(engine, dense, |dense, engine| {
        let mut output_data = vec![D::zero(); output_len];
        for (index, landing) in landings.iter().enumerate() {
            let matrix = matrices.get(index)?;
            let order = matrix.rows;
            if order == 0 {
                continue;
            }
            let output = &mut output_data[output_regions[landing.output].range()];
            apply(dense, engine, &mut state, matrix.data, order, output)?;
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

/// A square matrix function's validated routes: the canonical output
/// regions and length, the input's sector matrices and where each lands.
struct SquareSectorRoutes<'a, D> {
    output_regions: Arc<[CoupledSectorRegion]>,
    output_len: usize,
    matrices: InputMatricizations<'a, D>,
    landings: Vec<SectorLanding>,
}

/// The routes of a square matrix function of `input` into `output_space`,
/// which must preserve the input's hom space with canonical coupled-sector
/// storage; the input's row and column stackings must coincide.
fn square_sector_routes<'a, R, D: FactorScalar>(
    input: &BoundDynamicTensorRef<'a, R, D>,
    output_space: &BoundDynamicFusionMapSpace<R>,
) -> Result<SquareSectorRoutes<'a, D>, OperationError> {
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
    Ok(SquareSectorRoutes {
        output_regions,
        output_len,
        matrices,
        landings,
    })
}

/// The exponential of a Hermitian endomorphism (MatrixAlgebraKit
/// `exponential!` via `MatrixFunctionViaEigh`), every coupled sector of
/// nonzero order one input of a [`DenseFactorization::ExpViaEigh`]
/// destination batch writing the output region in place. One scope spans the
/// batch, and the first failure in sector order is the error.
///
/// The caller admits the endomorphism and its Hermiticity.
pub(crate) fn exp_via_eigh_dyn_into<E, R, D>(
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
    let SquareSectorRoutes {
        output_regions,
        output_len,
        matrices,
        landings,
    } = square_sector_routes(input, &output_space)?;
    let (_, blocks) = sector_blocks(&matrices)?;
    let solved: Vec<usize> = (0..blocks.len())
        .filter(|&index| blocks[index].1 > 0)
        .collect();
    let solved_blocks: Vec<_> = solved.iter().map(|&index| blocks[index]).collect();
    let mut layouts = Vec::new();
    let inputs = col_major_reads(&solved_blocks, &mut layouts)?;
    // Why allocator zeroing rather than uninitialized outputs: the
    // destinations are safe initialized views, and the zero pages of a large
    // output are first touched by the lane that writes them.
    let mut output_data = zeroed_payload::<D>(output_len);
    {
        let ranges: Vec<_> = landings
            .iter()
            .map(|landing| output_regions[landing.output].range())
            .collect();
        let mut slices = split_ranges(&mut output_data, &ranges)?;
        let dest_layouts: Vec<_> = solved_blocks
            .iter()
            .map(|&(_, order, _)| matrix_layout(order, order))
            .collect();
        let mut dests = solved
            .iter()
            .zip(&dest_layouts)
            .map(|(&index, layout)| {
                Ok(FactorDest::new(
                    [
                        Some(dest_write(std::mem::take(&mut slices[index]), layout)?),
                        None,
                        None,
                    ],
                    symtn_dense::FactorPlacement::AsComputed,
                ))
            })
            .collect::<Result<Vec<_>, OperationError>>()?;
        in_linalg_scope(engine, dense, |dense, engine| {
            factorize_batch_into_staged(
                engine,
                dense,
                DenseFactorization::ExpViaEigh,
                &inputs,
                &mut dests,
                |_, _, outcome| {
                    outcome.map_err(|failure| OperationError::Dense(failure.into_error()))?;
                    Ok(())
                },
            )
        })?;
    }

    BoundDynFactor::from_bound(
        output_space,
        output_data,
        source_space.nout(),
        source_space.nin(),
    )
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
