use super::*;
use crate::ContractError;

struct LoweredTypedNetwork<R> {
    ir: NetworkIR,
    infos: Vec<DenseTensorInfo>,
    spaces: Vec<Vec<GradedSpace<R>>>,
    rule_identity: RuleIdentity,
}

pub(crate) struct BoundSymmetricSlicedPlan<R> {
    plan: SymmetricSlicedPlan,
    occurrences: Vec<Vec<BoundSliceOccurrence>>,
    output_effective: Vec<GradedSpace<R>>,
    output_codomain_rank: usize,
}

#[derive(Clone, Copy)]
struct BoundSliceOccurrence {
    slice_index: usize,
    effective_axis: usize,
    partner: bool,
}

impl Network {
    /// Build and validate a network from written label lists.
    ///
    /// `inputs[i]` are operand `i`'s labels in flat leg order (codomain
    /// then domain of the tensor as passed, i.e. *before* any conj
    /// lowering), `conj[i]` marks adjoint operands, and `codomain_splits[i]`
    /// is the written `;` position, which every contraction and plan checks
    /// against the tensor's codomain rank before any plan lookup.
    /// `output_codomain_rank` is the output's `;` position (`None` for an
    /// all-codomain output). Labels are anything converting into
    /// [`TemporaryLabel`], such as `&str`, `char` or `TemporaryLabel`.
    /// Label structure (each label open-once or contracted-twice, output
    /// labels present and unique) is validated here.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError::MarkerCountMismatch`] for inconsistent operand
    /// marker counts, or [`ContractError::OutputCodomainRankOutOfRange`] when
    /// the output split exceeds its rank. Label validation returns
    /// [`ContractError::EmptyInput`], [`ContractError::DuplicateOutputLabel`],
    /// [`ContractError::UnknownOutputLabel`], [`ContractError::UnsupportedDiagonal`],
    /// [`ContractError::UnsupportedHyperedge`], [`ContractError::UnsupportedBatchLabel`],
    /// or [`ContractError::UnsupportedReduction`] unchanged.
    pub fn new<I, L>(
        inputs: I,
        conj: impl IntoIterator<Item = bool>,
        codomain_splits: impl IntoIterator<Item = usize>,
        output: impl IntoIterator<Item = L>,
        output_codomain_rank: Option<usize>,
    ) -> Result<Self, ContractError>
    where
        I: IntoIterator,
        I::Item: IntoIterator<Item = L>,
        L: Into<TemporaryLabel>,
    {
        let inputs: Vec<Vec<TemporaryLabel>> = inputs
            .into_iter()
            .map(|labels| labels.into_iter().map(Into::into).collect())
            .collect();
        let conj: Vec<bool> = conj.into_iter().collect();
        let codomain_splits: Vec<usize> = codomain_splits.into_iter().collect();
        let output: Vec<TemporaryLabel> = output.into_iter().map(Into::into).collect();
        if conj.len() != inputs.len() || codomain_splits.len() != inputs.len() {
            return Err(ContractError::MarkerCountMismatch {
                operands: inputs.len(),
                conj: conj.len(),
                codomain_splits: codomain_splits.len(),
            });
        }
        if let Some(k) = output_codomain_rank {
            if k > output.len() {
                return Err(ContractError::OutputCodomainRankOutOfRange {
                    codomain_rank: k,
                    output_rank: output.len(),
                });
            }
        }
        // Validates hyperedge structure (diagonal / hyperedge / batch /
        // reduction rejection) on the WRITTEN labels; conj rotation is a
        // cyclic per-operand relabeling that does not change the structure.
        NetworkIR::from_labels(inputs.clone(), output.clone())?;
        let mut first = HashMap::new();
        let contracted = inputs
            .iter()
            .enumerate()
            .map(|(operand, labels)| {
                labels
                    .iter()
                    .enumerate()
                    .map(|(axis, label)| match first.get(label) {
                        Some(&pair) => Some(pair),
                        None => {
                            first.insert(label, (operand, axis));
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let topology_hash = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (
                &inputs,
                &conj,
                &codomain_splits,
                &output,
                output_codomain_rank,
            )
                .hash(&mut hasher);
            hasher.finish()
        };
        Ok(Self {
            inputs,
            conj,
            codomain_splits,
            output,
            output_codomain_rank,
            contracted,
            topology_hash,
        })
    }

    /// Plans from storage-independent metadata of homogeneous typed Host
    /// tensors; payload storage is never read or transferred.
    pub fn plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        optimizer: &(impl DenseContractionOptimizer + ?Sized),
    ) -> Result<PlannedNetwork, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        self.plan_searched(tensors, |ir, cost| {
            ContractionPlan::from_steps(ir, optimizer.optimize(ir, cost)?)
        })
    }

    #[cfg(feature = "opt-path")]
    pub(crate) fn plan_with_optimizer_fallbacks<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        optimizer: &dyn DenseContractionOptimizer,
        fallbacks: &[&dyn DenseContractionOptimizer],
    ) -> Result<PlannedNetwork, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        self.plan_searched(tensors, |ir, cost| {
            let try_plan = |optimizer: &dyn DenseContractionOptimizer| {
                ContractionPlan::from_steps(ir, optimizer.optimize(ir, cost)?)
            };
            let mut result = try_plan(optimizer);
            for optimizer in fallbacks {
                if result.is_ok() {
                    break;
                }
                result = try_plan(*optimizer);
            }
            result
        })
    }

    /// Lowers `tensors` and runs `search` for a multi-operand network; a
    /// single operand has the empty order and is not searched.
    fn plan_searched<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        search: impl FnOnce(&NetworkIR, &DenseCostModel) -> crate::Result<ContractionPlan>,
    ) -> Result<PlannedNetwork, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let LoweredTypedNetwork { ir, infos, .. } = self.lower_typed(tensors)?;
        let plan = if ir.tensors().len() == 1 {
            ContractionPlan::from_steps(&ir, Vec::new()).map_err(invalid)?
        } else {
            let cost = DenseCostModel::from_network(&ir, &infos).map_err(invalid)?;
            search(&ir, &cost).map_err(invalid)?
        };
        self.finish_typed_plan(tensors, ir, plan)
    }

    /// Wraps an already searched structural order for typed execution.
    ///
    /// # Errors
    ///
    /// Returns an invalid-argument error when `plan` does not describe this
    /// network: its tensor count, output labels, or step result labels differ
    /// from the lowered inputs. This is the step-label check
    /// [`ContractionPlan::from_steps`] runs, repeated once per call.
    pub fn plan_with<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let LoweredTypedNetwork { ir, .. } = self.lower_typed(tensors)?;
        // The schedule compiler trusts its plan: an order for fewer tensors
        // would silently drop a closed scalar subnetwork.
        validate_contraction_plan_for_ir(&ir, &plan).map_err(invalid)?;
        self.finish_typed_plan(tensors, ir, plan)
    }

    /// Lowers planner-selected labels into a reconstructable coefficient-free plan.
    ///
    /// The result is unbound: it records self-consistent authority-leg snapshots
    /// but does not prove tensor provenance. [`Self::execute_symmetric_sliced`] binds
    /// it again against the actual typed tensors before execution.
    pub fn lower_symmetric_sliced_plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        sliced: SlicedPlan,
    ) -> std::result::Result<SymmetricSlicedPlan, SymmetricSliceLowerError<FacadeError<R>>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let lowered = self
            .lower_typed(tensors)
            .map_err(SymmetricSliceLowerError::Tensor)?;
        let legs = lowered
            .spaces
            .iter()
            .map(|spaces| {
                spaces
                    .iter()
                    .map(|space| __network::network_sector_leg(space).clone())
                    .collect()
            })
            .collect::<Vec<_>>();
        lower_symmetric_sliced_plan(&lowered.ir, lowered.rule_identity, &legs, sliced)
    }

    pub(crate) fn bind_symmetric_sliced_plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        plan: SymmetricSlicedPlan,
    ) -> std::result::Result<BoundSymmetricSlicedPlan<R>, SymmetricSliceLowerError<FacadeError<R>>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let lowered = self
            .lower_typed(tensors)
            .map_err(SymmetricSliceLowerError::Tensor)?;
        validate_contraction_plan_for_ir(&lowered.ir, plan.plan())
            .map_err(SymmetricSliceLowerError::InvalidPlan)?;
        let expected = plan.slices().rule_identity().clone();
        if expected != lowered.rule_identity {
            return Err(SymmetricSliceLowerError::RuleMismatch {
                expected,
                actual: lowered.rule_identity,
            });
        }

        let mut specs = Vec::with_capacity(plan.slices().indices().len());
        for index in plan.slices().indices() {
            let edge = lowered.ir.edge(index.label()).ok_or_else(|| {
                SymmetricSliceLowerError::InvalidSlice(SliceError::UnknownLabel(
                    index.label().clone(),
                ))
            })?;
            let authority = index.authority();
            let expected_authority = edge.occurrences()[0];
            if authority != expected_authority {
                return Err(SymmetricSliceLowerError::InvalidSlice(
                    SliceError::InvalidAuthority {
                        label: index.label().clone(),
                        expected: expected_authority,
                        actual: authority,
                    },
                ));
            }
            let actual_leg = lowered
                .spaces
                .get(authority.tensor().index())
                .and_then(|spaces| spaces.get(authority.axis()))
                .map(|space| __network::network_sector_leg(space).clone())
                .ok_or_else(|| SymmetricSliceLowerError::MissingAuthority {
                    label: index.label().clone(),
                    authority,
                })?;
            if index.authority_leg() != &actual_leg {
                return Err(SymmetricSliceLowerError::AuthorityLegMismatch {
                    label: index.label().clone(),
                    authority,
                    expected: actual_leg,
                    actual: index.authority_leg().clone(),
                });
            }
            specs.push(SymmetricSliceSpec::new(
                index.label().clone(),
                authority,
                actual_leg,
                index.pieces().to_vec(),
            ));
        }
        let rebound = SymmetricSlicePlan::try_new(&lowered.ir, lowered.rule_identity, specs)
            .map_err(SymmetricSliceLowerError::InvalidSlice)?;
        debug_assert_eq!(&rebound, plan.slices());
        let mut occurrences = vec![Vec::new(); tensors.len()];
        for (slice_index, index) in rebound.indices().iter().enumerate() {
            let edge = lowered
                .ir
                .edge(index.label())
                .expect("rebound label exists");
            for (occurrence_index, occurrence) in edge.occurrences().iter().enumerate() {
                occurrences[occurrence.tensor().index()].push(BoundSliceOccurrence {
                    slice_index,
                    effective_axis: occurrence.axis(),
                    partner: occurrence_index != 0,
                });
            }
        }
        let output_effective = self
            .output
            .iter()
            .map(|label| {
                let occurrence = lowered
                    .ir
                    .edge(label)
                    .expect("validated output label exists")
                    .occurrences()[0];
                lowered.spaces[occurrence.tensor().index()][occurrence.axis()].clone()
            })
            .collect();
        let output_codomain_rank = self.output_codomain_rank.unwrap_or(self.output.len());
        let plan = SymmetricSlicedPlan::new(plan.plan().clone(), rebound);
        Ok(BoundSymmetricSlicedPlan {
            plan,
            occurrences,
            output_effective,
            output_codomain_rank,
        })
    }

    /// Executes coefficient-free symmetric slices just in time and returns the
    /// result with named network-owned payload statistics.
    ///
    /// Counted payloads are compact sliced inputs, live planned contraction
    /// and permutation destinations, the current partial, the private
    /// accumulator, and path-owned dense buffers. Small metadata, external
    /// backend scratch, and symtn Runtime context scratch are excluded. The
    /// latter includes the pooled output-sized `copyC` temporary used by
    /// split-moving and in-split reorder contractions, so
    /// [`SymmetricSliceStats::peak_total_bytes`] can be below the process peak
    /// by that temporary's capacity. The ceiling is checked against observed
    /// payloads after each tensor kernel returns and before publication; it is
    /// not an allocator reservation or an out-of-memory prevention guarantee.
    /// At every observation, total payload is checked as the full destination
    /// bytes plus non-destination workspace bytes. A successful return proves
    /// that [`SymmetricSliceStats::peak_total_bytes`] did not exceed the total
    /// ceiling.
    #[expect(
        clippy::type_complexity,
        reason = "the public sliced execution API returns its tensor and measured statistics together"
    )]
    pub fn execute_symmetric_sliced<R, D>(
        &self,
        tensors: &[&TensorMap<R, D>],
        plan: SymmetricSlicedPlan,
        measured_payload_ceiling: usize,
    ) -> std::result::Result<
        (TensorMap<R, D>, SymmetricSliceStats),
        SymmetricSliceExecutionError<FacadeError<R>>,
    >
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
    {
        // Slicing reads each input once per slice; a compact diagonal would be
        // densified operation-locally on every one of those reads. Reject it
        // before restriction/provider work until that copy is accounted for
        // once per execution.
        if tensors
            .iter()
            .any(|tensor| __network::network_has_compact_payload(tensor))
        {
            return Err(SymmetricSliceExecutionError::Tensor(
                invalid("symmetric sliced execution requires dense Host payloads").into(),
            ));
        }
        reject_non_symmetric_network(tensors, self.inputs.len() > 1)
            .map_err(SymmetricSliceExecutionError::Tensor)?;

        let bound = self
            .bind_symmetric_sliced_plan(tensors, plan)
            .map_err(SymmetricSliceExecutionError::Bind)?;
        let planned = self
            .plan_with(tensors, bound.plan.plan().clone())
            .map_err(SymmetricSliceExecutionError::Tensor)?;
        let mut workspace = NetworkExecutionWorkspace::default();
        let (codomain, domain) = bound.output_effective.split_at(bound.output_codomain_rank);
        let authority_input_slot = planned
            .schedule
            .steps
            .last()
            .map_or(0, |step| step.authority_input_slot);
        let mut accumulator = __network::network_zeros_from_effective_legs(
            tensors[authority_input_slot],
            codomain,
            domain,
        )
        .map_err(SymmetricSliceExecutionError::Tensor)?;
        let mut meter =
            PayloadMeter::new(measured_payload_ceiling, &accumulator).map_err(map_payload_error)?;
        if bound.plan.slices().nslices() == 0 {
            meter
                .observe::<R, D, Vec<D>>(&[], &[], &[])
                .map_err(map_payload_error)?;
            return Ok((accumulator, meter.stats()));
        }

        for ordinal in 0..bound.plan.slices().nslices() {
            let selected = bound
                .plan
                .slices()
                .combination(ordinal)
                .expect("ordinal is bounded by semantic nslices");
            let mut scatter_ranges = vec![None; bound.output_effective.len()];
            for (index, piece) in bound.plan.slices().indices().iter().zip(&selected) {
                if let Some(output_position) = index.output_position() {
                    let range = piece.range();
                    scatter_ranges[output_position] = Some(range.start()..range.end());
                }
            }
            let mut owned = Vec::with_capacity(tensors.len());
            let mut compact_flags = Vec::with_capacity(tensors.len());
            for (operand, tensor) in tensors.iter().enumerate() {
                let occurrences = &bound.occurrences[operand];
                compact_flags.push(!occurrences.is_empty());
                if occurrences.is_empty() {
                    owned.push((*tensor).clone());
                    continue;
                }
                let restrictions = occurrences
                    .iter()
                    .map(|occurrence| {
                        let piece = selected[occurrence.slice_index];
                        let range = piece.range();
                        NetworkDegeneracyRestriction {
                            effective_axis: occurrence.effective_axis,
                            authority_sector: piece.sector(),
                            range: range.start()..range.end(),
                            partner: occurrence.partner,
                        }
                    })
                    .collect::<Vec<_>>();
                owned.push(
                    __network::network_restrict_degeneracies(
                        tensor,
                        self.conj[operand],
                        &restrictions,
                    )
                    .map_err(SymmetricSliceExecutionError::Tensor)?,
                );
            }
            let compact_inputs = owned
                .iter()
                .zip(&compact_flags)
                .filter_map(|(tensor, &compact)| compact.then_some(tensor));
            meter.set_base(compact_inputs);
            let payloads = intermediate_payloads(&workspace.intermediates);
            meter
                .observe(&workspace.slots, &workspace.producers, &payloads)
                .map_err(map_payload_error)?;

            let refs = owned.iter().collect::<Vec<_>>();
            let partial = planned
                .execute_with_workspace_meter(&refs, &mut workspace, Some(&mut meter))
                .map_err(map_metered_network_error)?;
            #[cfg(test)]
            crate::probe::hit(crate::probe::Event::SliceJobCompleted);

            let compact_inputs = owned
                .iter()
                .zip(&compact_flags)
                .filter_map(|(tensor, &compact)| compact.then_some(tensor));
            meter.set_base(compact_inputs.chain(std::iter::once(&partial)));
            let payloads = intermediate_payloads(&workspace.intermediates);
            meter
                .observe(&workspace.slots, &workspace.producers, &payloads)
                .map_err(map_payload_error)?;

            __network::network_scatter_add_assign(&mut accumulator, &partial, &scatter_ranges)
                .map_err(|error| {
                    SymmetricSliceExecutionError::Tensor(FacadeError::<R>::from(error))
                })?;
        }
        Ok((accumulator, meter.stats()))
    }

    fn finish_typed_plan<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
        ir: NetworkIR,
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        let input_codomain_ranks = tensors
            .iter()
            .map(|tensor| tensor.codomain_rank())
            .collect();
        let lowered_codomain_ranks = tensors
            .iter()
            .enumerate()
            .map(|(i, tensor)| {
                if self.conj[i] {
                    tensor.domain_rank()
                } else {
                    tensor.codomain_rank()
                }
            })
            .collect::<Vec<_>>();
        self.finish_plan(input_codomain_ranks, lowered_codomain_ranks, ir, plan)
            .map_err(Into::into)
    }

    fn finish_plan(
        &self,
        input_codomain_ranks: Vec<usize>,
        lowered_codomain_ranks: Vec<usize>,
        ir: NetworkIR,
        plan: ContractionPlan,
    ) -> Result<PlannedNetwork, Error> {
        let schedule = compile_schedule(
            &ir,
            &plan,
            self.output_codomain_rank,
            &lowered_codomain_ranks,
        )?;
        Ok(PlannedNetwork {
            owner_token: NEXT_PLAN_OWNER_TOKEN.fetch_add(1, Ordering::Relaxed),
            plan,
            conj: self.conj.clone(),
            input_codomain_ranks,
            schedule,
        })
    }

    fn lower_typed<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
    ) -> Result<LoweredTypedNetwork<R>, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        if tensors.len() != self.inputs.len() {
            return Err(invalid(format!(
                "network has {} operands but {} tensors were given",
                self.inputs.len(),
                tensors.len()
            ))
            .into());
        }
        let Some(rule_identity) = typed_operand_identity(tensors)? else {
            unreachable!("a validated Network has at least one operand")
        };

        network_operand_preflight(
            tensors,
            &self.inputs,
            &self.conj,
            &self.codomain_splits,
            &self.contracted,
        )?;
        let mut lowered_labels = Vec::with_capacity(tensors.len());
        let mut infos = Vec::with_capacity(tensors.len());
        let mut lowered_spaces = Vec::with_capacity(tensors.len());
        for ((&tensor, labels), &conj) in tensors.iter().zip(&self.inputs).zip(&self.conj) {
            let dims = <R::Mode as NetworkModeDispatch<R, D>>::leg_dims(tensor)?;
            let split = tensor.codomain_rank();
            if conj {
                lowered_labels.push(rotate(labels, split));
                infos.push(DenseTensorInfo::new(rotate(&dims, split)));
            } else {
                lowered_labels.push(labels.clone());
                infos.push(DenseTensorInfo::new(dims));
            }
            lowered_spaces.push(typed_effective_spaces(tensor, conj)?);
        }
        let ir = NetworkIR::from_labels(lowered_labels, self.output.clone())
            .map_err(|error| FacadeError::<R>::from(invalid(error)))?;
        Ok(LoweredTypedNetwork {
            ir,
            infos,
            spaces: lowered_spaces,
            rule_identity,
        })
    }
}
