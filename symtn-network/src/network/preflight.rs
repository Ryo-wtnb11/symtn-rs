use super::*;

/// Runtime and rule identity of a network's operands, the first check of
/// every lowering and preflight; `None` for no operand.
pub(super) fn typed_operand_identity<R, D, S>(
    tensors: &[&TensorMap<R, D, S>],
) -> Result<Option<RuleIdentity>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let Some(first) = tensors.first() else {
        return Ok(None);
    };
    let runtime = __network::runtime_identity(first.runtime());
    let identity = TypedSectorAdmission::typed_rule_identity(first.provider());
    for tensor in tensors.iter().skip(1) {
        if !runtime.matches(tensor.runtime()) {
            return Err(Error::RuntimeMismatch.into());
        }
        if identity != TypedSectorAdmission::typed_rule_identity(tensor.provider()) {
            return Err(Error::RuleMismatch.into());
        }
    }
    Ok(Some(identity))
}

/// The metadata preflight of a network, run by [`Network::contract`] before
/// the plan lookup, lease or transfer and by every typed lowering: each
/// operand's written rank and `;` split, then every contracted leg against the
/// dual of its partner. The caller decides braiding and on the device
/// placement first.
///
/// Allocation-free, because a warm plan-cache hit runs it on every call and
/// the cache key holds no sectors: legs are borrowed from the operands and
/// compared through the dual map, and no label map is built.
///
/// `contracted` is the pairing [`Network::new`] resolved once (its
/// `contracted` field): for each operand and written axis, the earlier
/// `(operand, written axis)` with the same label. It makes the pass O(N) in
/// the total legs N, where rediscovering each partner by a scan would be
/// O(N²). `Network::new` rejects a label written twice on one operand, so
/// every pair spans two operands.
pub(super) fn network_operand_preflight<R, D, S, L, C>(
    tensors: &[&TensorMap<R, D, S>],
    inputs: &[impl AsRef<[L]>],
    conj: &[bool],
    splits: &[usize],
    contracted: &[C],
) -> Result<(), FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar,
    S: TensorStorage<D>,
    L: PartialEq + std::fmt::Display,
    C: AsRef<[Option<(usize, usize)>]>,
{
    typed_operand_identity(tensors)?;
    for (index, &tensor) in tensors.iter().enumerate() {
        let labels = inputs[index].as_ref();
        if labels.len() != tensor.rank() {
            return Err(invalid(format!(
                "operand {index} has {} labels but tensor rank {}",
                labels.len(),
                tensor.rank()
            ))
            .into());
        }
        let split = splits[index];
        if split != tensor.codomain_rank() {
            return Err(invalid(format!(
                "operand {index} puts {split} label(s) before `;` but the tensor's codomain rank is {}",
                tensor.codomain_rank()
            ))
            .into());
        }
    }
    let Some(first) = tensors.first() else {
        return Ok(());
    };
    // A lowered axis and its written axis are the same, except on a `conj`
    // operand, which is read through its adjoint: lowered axis `axis` is
    // written axis `(axis + codomain_rank) % rank`.
    let written_axis = |operand: usize, axis: usize| {
        let tensor = tensors[operand];
        if conj[operand] {
            (axis + tensor.codomain_rank()) % tensor.rank()
        } else {
            axis
        }
    };
    let lowered_axis = |operand: usize, written: usize| {
        let tensor = tensors[operand];
        if conj[operand] {
            (written + tensor.rank() - tensor.codomain_rank()) % tensor.rank()
        } else {
            written
        }
    };
    let label =
        |operand: usize, axis: usize| &inputs[operand].as_ref()[written_axis(operand, axis)];
    // Lowered axis `axis` of `operand`: the stored leg, and whether the
    // lowered leg is that leg's dual.
    let leg = |operand: usize, axis: usize| {
        let tensor = tensors[operand];
        let adjoint = conj[operand];
        let codomain_rank = tensor.codomain_rank();
        let source = if adjoint {
            (axis + codomain_rank) % tensor.rank()
        } else {
            axis
        };
        let stored = __network::network_source_leg(tensor, source).ok_or_else(|| {
            FacadeError::<R>::from(invalid(format!("operand {operand} has no leg")))
        })?;
        Ok::<_, FacadeError<R>>((stored, (source >= codomain_rank) != adjoint))
    };
    // The cross-check of the pairing in debug builds: the partner a scan over
    // the earlier axes finds.
    let scan = |operand: usize, axis: usize| {
        let written = label(operand, axis);
        (0..=operand)
            .flat_map(|previous| {
                let end = if previous == operand {
                    axis
                } else {
                    tensors[previous].rank()
                };
                (0..end).map(move |previous_axis| (previous, previous_axis))
            })
            .find(|&(previous, previous_axis)| label(previous, previous_axis) == written)
    };
    for operand in 0..tensors.len() {
        for axis in 0..tensors[operand].rank() {
            let pair = contracted[operand].as_ref()[written_axis(operand, axis)]
                .map(|(previous, written)| (previous, lowered_axis(previous, written)));
            debug_assert!(
                pair == scan(operand, axis),
                "pairing {pair:?} of operand {operand} lowered axis {axis} \
                 disagrees with the label scan {:?}",
                scan(operand, axis)
            );
            let Some((previous_operand, previous_axis)) = pair else {
                continue;
            };
            let written = label(operand, axis);
            if !legs_contract(
                first.provider(),
                leg(previous_operand, previous_axis)?,
                leg(operand, axis)?,
            )? {
                return Err(invalid(format!(
                    "space mismatch for contracted label `{written}` between operand {previous_operand} leg {previous_axis} and operand {operand} leg {axis}"
                ))
                .into());
            }
        }
    }
    Ok(())
}

/// Whether two lowered legs, each a stored leg and whether it is read as that
/// leg's dual, may be contracted: the second must be the dual of the first,
/// as `GradedSpace::try_dual` would build it. Only the parity of the two
/// dualisations matters: when exactly one side is dualised the stored legs
/// must be equal, which is TensorKit's `space(A, i) == space(B, j)'` with a
/// structural dual.
pub(super) fn legs_contract<R>(
    provider: &R,
    (lhs, lhs_dualised): (&SectorLeg, bool),
    (rhs, rhs_dualised): (&SectorLeg, bool),
) -> Result<bool, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    if lhs_dualised != rhs_dualised {
        return Ok(lhs == rhs);
    }
    is_dual_of(provider, lhs, rhs)
}

/// `into == dual(from)` for sorted, duplicate-free legs: the opposite duality
/// flag, as many sectors, and the dual of every sector of `from` in `into`
/// with the same degeneracy. The round trip `dual(dual(s)) == s` proves the
/// dual injective on `from`, so its image fills `into` exactly; a provider
/// whose dual is not an involution is rejected with the error
/// `GradedSpace::try_dual` raises for a non-injective dual, not accepted.
fn is_dual_of<R>(provider: &R, from: &SectorLeg, into: &SectorLeg) -> Result<bool, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
{
    if from.is_dual() == into.is_dual() || from.sectors().len() != into.sectors().len() {
        return Ok(false);
    }
    let dual_of = |sector| {
        TypedSectorAdmission::try_dual_id(provider, sector)
            .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)
    };
    for (&sector, &degeneracy) in from.sectors().iter().zip(from.degeneracies()) {
        let dual = dual_of(sector)?;
        if dual_of(dual)? != sector {
            return Err(invalid(format!(
                "dual map is not injective: sector {dual:?} appears multiple times"
            ))
            .into());
        }
        match into.sectors().binary_search(&dual) {
            Ok(position) if into.degeneracies()[position] == degeneracy => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// The typed `contract`'s braiding boundary (TensorKit `blas_contract!`
/// requires symmetric braiding), decided once for a network whose schedule
/// contracts: the same error the first contraction step would raise, without
/// the permutes, trace pre-step, sliced accumulator or plan lookup that would
/// otherwise run before it.
pub(super) fn reject_non_symmetric_network<R, D, S>(
    tensors: &[&TensorMap<R, D, S>],
    contracts: bool,
) -> Result<(), FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    let Some(first) = tensors.first().filter(|_| contracts) else {
        return Ok(());
    };
    let braiding = <R::Mode as NetworkModeDispatch<R, D, S>>::braiding_style(first.provider());
    symtn::typed::reject_non_symmetric_contraction(braiding)
        .map_err(|error| FacadeError::<R>::from(Error::from(error)))
}

pub(super) fn validate_typed_contracted_pairs<R, D, S>(
    tensors: &[TensorMap<R, D, S>],
    pairs: &[InputLegPair],
) -> Result<(), FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, S>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    let spaces = tensors
        .iter()
        .map(typed_flat_spaces)
        .collect::<Result<Vec<_>, _>>()?;
    for &((lhs_slot, lhs_axis), (rhs_slot, rhs_axis)) in pairs {
        if spaces[rhs_slot][rhs_axis] != spaces[lhs_slot][lhs_axis].try_dual()? {
            return Err(invalid(format!(
                "contracted input spaces mismatch between operand {lhs_slot} leg {lhs_axis} and operand {rhs_slot} leg {rhs_axis}"
            ))
            .into());
        }
    }
    Ok(())
}

fn typed_flat_spaces<R, D, S>(
    tensor: &TensorMap<R, D, S>,
) -> Result<Vec<GradedSpace<R>>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    let mut spaces = tensor.codomain();
    spaces.extend(
        tensor
            .domain()
            .iter()
            .map(GradedSpace::try_dual)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(spaces)
}

pub(super) fn typed_effective_spaces<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    adjoint: bool,
) -> Result<Vec<GradedSpace<R>>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorModeDispatch<R>,
    D: TensorScalar,
    S: TensorStorage<D>,
{
    if !adjoint {
        return typed_flat_spaces(tensor);
    }
    let mut spaces = tensor.domain();
    spaces.extend(
        tensor
            .codomain()
            .iter()
            .map(GradedSpace::try_dual)
            .collect::<Result<Vec<_>, _>>()?,
    );
    Ok(spaces)
}

pub(super) fn rotate<T: Clone>(items: &[T], split: usize) -> Vec<T> {
    items[split..]
        .iter()
        .chain(items[..split].iter())
        .cloned()
        .collect()
}
