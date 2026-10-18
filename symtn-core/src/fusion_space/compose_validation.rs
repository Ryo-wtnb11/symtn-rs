use super::*;

pub(super) fn dual_sector_leg<R>(rule: &R, leg: &SectorLeg) -> SectorLeg
where
    R: FusionRule,
{
    leg.dual(rule)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn tensorcontract_descriptor<'a>(
    lhs: OrientedFusionTreeHomSpace<'a>,
    rhs: OrientedFusionTreeHomSpace<'a>,
    lhs_contracting_axes: &[usize],
    rhs_contracting_axes: &[usize],
    output_axes: &[usize],
    dst_codomain_rank: usize,
) -> Result<HomSpaceDescriptor<'a>, CoreError> {
    if lhs_contracting_axes.len() != rhs_contracting_axes.len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_contracting_axes.len(),
            actual: rhs_contracting_axes.len(),
        });
    }

    let lhs_seen = validate_axis_subset_inline("lhs", lhs_contracting_axes, lhs.rank())?;
    let rhs_seen = validate_axis_subset_inline("rhs", rhs_contracting_axes, rhs.rank())?;
    let lhs_open_axes = lhs_seen.complement().collect::<SmallVec<[usize; 8]>>();
    let rhs_open_axes = rhs_seen.complement().collect::<SmallVec<[usize; 8]>>();
    let output_rank = lhs_open_axes.len() + rhs_open_axes.len();
    validate_permutation_inline(output_axes, output_rank)?;
    if dst_codomain_rank > output_rank {
        return Err(CoreError::StructureRankMismatch {
            expected: output_rank,
            actual: dst_codomain_rank,
        });
    }

    let mut open_legs = SmallVec::<[OrientedLegView<'a>; 8]>::new();
    open_legs.extend(lhs_open_axes.iter().map(|&axis| {
        lhs.external_axis_leg_view(axis)
            .expect("validated axis belongs to the lhs")
    }));
    open_legs.extend(rhs_open_axes.iter().map(|&axis| {
        rhs.external_axis_leg_view(axis)
            .expect("validated axis belongs to the rhs")
    }));
    // Why not materialize two permuted operands and their composition: output
    // ordering observes only these final external views, and doing so would
    // repeat orientation arithmetic before the final HomSpace exists.
    let descriptor = HomSpaceDescriptor::new(
        output_axes[..dst_codomain_rank]
            .iter()
            .map(|&axis| open_legs[axis]),
        output_axes[dst_codomain_rank..]
            .iter()
            .map(|&axis| open_legs[axis].toggled()),
    );
    Ok(descriptor)
}

fn validate_axis_subset_inline(
    tensor: &'static str,
    axes: &[usize],
    rank: usize,
) -> Result<crate::axes::AxisMask, CoreError> {
    crate::axes::validate_axis_subset(axes, rank).map_err(|_| CoreError::InvalidAxisSet {
        tensor,
        axes: axes.to_vec(),
        rank,
    })
}

pub(crate) fn validate_permutation_inline(
    permutation: &[usize],
    rank: usize,
) -> Result<(), CoreError> {
    crate::axes::validate_permutation(permutation, rank).map_err(|_| {
        CoreError::InvalidPermutation {
            permutation: permutation.to_vec(),
            rank,
        }
    })
}

pub(super) fn validate_axis_selection(
    codomain_axes: &[usize],
    domain_axes: &[usize],
    rank: usize,
) -> Result<(), CoreError> {
    for &axis in codomain_axes.iter().chain(domain_axes) {
        if axis >= rank {
            let mut axes = Vec::with_capacity(codomain_axes.len() + domain_axes.len());
            axes.extend_from_slice(codomain_axes);
            axes.extend_from_slice(domain_axes);
            return Err(CoreError::InvalidPermutation {
                permutation: axes,
                rank,
            });
        }
    }
    Ok(())
}

/// `axes` are the external axes `(lhs, rhs)` being contracted; the error
/// reports each operand's external-axis flag, which is what callers see.
fn contracted_leg_duality_mismatch(
    (lhs_axis, rhs_axis): (usize, usize),
    lhs_domain_is_dual: bool,
    rhs_codomain_is_dual: bool,
) -> CoreError {
    CoreError::ContractedLegDualityMismatch {
        lhs_axis,
        rhs_axis,
        lhs_is_dual: !lhs_domain_is_dual,
        rhs_is_dual: rhs_codomain_is_dual,
    }
}

pub(crate) fn validate_composed_leg(
    lhs_domain: &SectorLeg,
    rhs_codomain: &SectorLeg,
    axes: (usize, usize),
) -> Result<(), CoreError> {
    if lhs_domain.is_dual() != rhs_codomain.is_dual() {
        return Err(contracted_leg_duality_mismatch(
            axes,
            lhs_domain.is_dual(),
            rhs_codomain.is_dual(),
        ));
    }
    // TensorKit parity: `A * B` requires `domain(A) == codomain(B)` as
    // spaces, so the stored legs must match verbatim (domain legs store the
    // domain space's own sectors; verified against TensorKit v0.16:
    // `rand(V ← V) * rand(V ← V)` works for V = U1Space(0=>1, 1=>1), a
    // sector set that is not dualization-closed, while `(V ← V) * (? ← V')`
    // is a SpaceMismatch).
    if lhs_domain.sectors().len() != rhs_codomain.sectors().len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_domain.sectors().len(),
            actual: rhs_codomain.sectors().len(),
        });
    }
    for ((expected, expected_deg), (actual, actual_deg)) in
        lhs_domain.iter().zip(rhs_codomain.iter())
    {
        if expected != actual {
            return Err(CoreError::SectorMismatch { expected, actual });
        }
        if expected_deg != actual_deg {
            return Err(CoreError::LegDegeneracyMismatch {
                sector: expected,
                expected: expected_deg,
                actual: actual_deg,
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_oriented_composed_leg<R>(
    rule: &R,
    lhs_domain: OrientedLegView<'_>,
    rhs_codomain: OrientedLegView<'_>,
    axes: (usize, usize),
) -> Result<(), CoreError>
where
    R: FusionRule,
{
    let valid = lhs_domain.is_dual() == rhs_codomain.is_dual()
        && lhs_domain.source.sectors().len() == rhs_codomain.source.sectors().len()
        && lhs_domain.source.iter().all(|(sector, degeneracy)| {
            let expected = lhs_domain.mapped_sector(rule, sector);
            let rhs_source_sector = if rhs_codomain.dualize {
                rule.dual(expected)
            } else {
                expected
            };
            rhs_codomain.source.degeneracy(rhs_source_sector) == Some(degeneracy)
        });
    if valid {
        return Ok(());
    }
    // Report the exact error variant and first mismatching sector only
    // on an invalid request. The valid hot path never sorts or rebuilds a leg.
    validate_composed_leg(
        &lhs_domain.materialize(rule),
        &rhs_codomain.materialize(rule),
        axes,
    )
}

pub(super) fn validate_oriented_composed_leg_checked<R>(
    rule: &R,
    lhs_domain: OrientedLegView<'_>,
    rhs_codomain: OrientedLegView<'_>,
    axes: (usize, usize),
) -> Result<(), CheckedFusionSpaceError>
where
    R: CheckedFusionAlgebra,
{
    if lhs_domain.is_dual() != rhs_codomain.is_dual() {
        return Err(contracted_leg_duality_mismatch(
            axes,
            lhs_domain.is_dual(),
            rhs_codomain.is_dual(),
        )
        .into());
    }
    if lhs_domain.source.sectors().len() != rhs_codomain.source.sectors().len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_domain.source.sectors().len(),
            actual: rhs_codomain.source.sectors().len(),
        }
        .into());
    }
    let mut valid = true;
    for (sector, degeneracy) in lhs_domain.source.iter() {
        let expected = lhs_domain.try_mapped_sector(rule, sector)?;
        let rhs_source_sector = if rhs_codomain.dualize {
            rule.try_dual_sector(expected)?
        } else {
            expected
        };
        if rhs_codomain.source.degeneracy(rhs_source_sector) != Some(degeneracy) {
            valid = false;
            break;
        }
    }
    if valid {
        return Ok(());
    }
    // Why not fall back to the infallible materializer: malformed-space error
    // detail must not turn a representability failure into an unwind.
    validate_composed_leg(
        &lhs_domain.try_materialize(rule)?,
        &rhs_codomain.try_materialize(rule)?,
        axes,
    )
    .map_err(Into::into)
}

pub(super) fn validate_oriented_composed_leg_generic_checked<R>(
    rule: &R,
    lhs_domain: OrientedLegView<'_>,
    rhs_codomain: OrientedLegView<'_>,
    axes: (usize, usize),
) -> Result<(), CheckedGenericStructureError<R::Error>>
where
    R: CheckedGenericFusion,
{
    if lhs_domain.is_dual() != rhs_codomain.is_dual() {
        return Err(contracted_leg_duality_mismatch(
            axes,
            lhs_domain.is_dual(),
            rhs_codomain.is_dual(),
        )
        .into());
    }
    if lhs_domain.source.sectors().len() != rhs_codomain.source.sectors().len() {
        return Err(CoreError::DimensionMismatch {
            expected: lhs_domain.source.sectors().len(),
            actual: rhs_codomain.source.sectors().len(),
        }
        .into());
    }
    let lhs = lhs_domain.try_materialize_generic(rule)?;
    let rhs = rhs_codomain.try_materialize_generic(rule)?;
    validate_composed_leg(&lhs, &rhs, axes).map_err(Into::into)
}
