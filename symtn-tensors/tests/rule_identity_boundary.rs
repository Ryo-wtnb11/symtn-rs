use std::sync::Arc;
use symtn_core::{
    BlockStructure, BraidingStyleKind, CoreError, FusionRule, FusionStyleKind, FusionTreeHomSpace,
    MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    RuleIdentity, SU2FusionRule, SectorId, SectorVec, U1FusionRule, Z2FusionRule,
};
use symtn_tensors::{
    tensorcontract_fusion_block_specs, BoundDynamicFusionMapSpace, ContractDestinationInit,
    FusionOperand, TensorContractFusionExecutionContext, TensorContractSpec,
    TreeTransformRuleCacheKey,
};
use symtn_tensors::{DynamicFusionMapSpace, OperationError};

#[derive(Clone)]
struct SymbolOnlyRule {
    identity: RuleIdentity,
    symbol: f64,
}

#[test]
fn su2_tree_transform_key_contains_the_rule_identity() {
    let rule = SU2FusionRule;
    assert_eq!(rule.tree_transform_rule_cache_key(), rule.rule_identity());
}

impl SymbolOnlyRule {
    fn new(symbol: f64) -> Self {
        Self {
            identity: RuleIdentity::new_unique::<Self>(),
            symbol,
        }
    }
}

impl FusionRule for SymbolOnlyRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        [SectorId::new(0)].into_iter().collect()
    }
}

impl MultiplicityFreeFusionRule for SymbolOnlyRule {}

impl MultiplicityFreeFusionSymbols for SymbolOnlyRule {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        self.symbol
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        self.symbol
    }
}

impl MultiplicityFreeRigidSymbols for SymbolOnlyRule {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

fn bound_scalar_space<R>(rule: &R) -> DynamicFusionMapSpace
where
    R: MultiplicityFreeRigidSymbols,
{
    DynamicFusionMapSpace::from_final_homspace(rule, FusionTreeHomSpace::from_sector_ids([], []))
        .unwrap()
}

fn unbound_scalar_space<R>(rule: &R) -> DynamicFusionMapSpace
where
    R: MultiplicityFreeRigidSymbols,
{
    let bound = bound_scalar_space(rule);
    unbound_scalar_space_like(&bound)
}

fn unbound_scalar_space_like(bound: &DynamicFusionMapSpace) -> DynamicFusionMapSpace {
    DynamicFusionMapSpace::try_from_layout(
        bound.homspace().clone(),
        BlockStructure::clone(bound.structure()),
    )
    .unwrap()
}

/// Binds a typed scalar space to `rule` at the dynamic ingress, the one
/// place a contraction's provider meets its spaces.
fn bind<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<BoundDynamicFusionMapSpace<R>, OperationError>
where
    R: MultiplicityFreeFusionRule + Clone,
{
    BoundDynamicFusionMapSpace::bind_multiplicity_free((*space).clone(), Arc::new(rule.clone()))
}

/// The scalar contraction `dst = 1 * lhs * rhs` with every space bound to
/// `rule`: the error of the first rejected binding, else of the contraction.
fn scalar_contract_error<R>(
    rule: &R,
    dst_space: DynamicFusionMapSpace,
    lhs_space: DynamicFusionMapSpace,
    rhs_space: DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> OperationError
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Clone,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
{
    let bound = (|| {
        Ok::<_, OperationError>((
            bind(rule, &dst_space)?,
            bind(rule, &lhs_space)?,
            bind(rule, &rhs_space)?,
        ))
    })();
    let (dst, lhs, rhs) = match bound {
        Ok(bound) => bound,
        Err(error) => return error,
    };
    TensorContractFusionExecutionContext::<f64, R::Key>::default()
        .tensorcontract_planned_into(
            &dst,
            &mut [0.0],
            (FusionOperand::direct(lhs.space()), &[2.0]),
            (FusionOperand::direct(rhs.space()), &[3.0]),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &[],
            1.0,
            ContractDestinationInit::Axpby(0.0),
        )
        .unwrap_err()
}

fn assert_missing_rule_identity(error: OperationError) {
    assert!(matches!(
        error,
        OperationError::Core(CoreError::MissingFusionRuleIdentity)
    ));
}

fn assert_rule_mismatch(error: OperationError) {
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn bound_space_rejects_rebinding_to_same_type_with_different_semantics() {
    let first = SymbolOnlyRule::new(1.0);
    let second = SymbolOnlyRule::new(-1.0);
    let space = unbound_scalar_space(&first).try_bind_rule(&first).unwrap();

    assert!(matches!(
        space.try_bind_rule(&second),
        Err(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn fusion_contract_block_specs_rejects_each_unbound_operand() {
    let rule = Z2FusionRule;
    let bound = || bound_scalar_space(&rule);
    let unbound = || unbound_scalar_space(&rule);
    let axes = TensorContractSpec::with_default_output_order(&[], &[]);

    assert_missing_rule_identity(
        tensorcontract_fusion_block_specs(&rule, &unbound(), &bound(), &bound(), axes).unwrap_err(),
    );
    assert_missing_rule_identity(
        tensorcontract_fusion_block_specs(&rule, &bound(), &unbound(), &bound(), axes).unwrap_err(),
    );
    assert_missing_rule_identity(
        tensorcontract_fusion_block_specs(&rule, &bound(), &bound(), &unbound(), axes).unwrap_err(),
    );
}

#[test]
fn fusion_contract_rejects_z2_spaces_with_u1_despite_overlapping_sector_ids() {
    let z2 = Z2FusionRule;
    let error = scalar_contract_error(
        &U1FusionRule,
        bound_scalar_space(&z2),
        bound_scalar_space(&z2),
        bound_scalar_space(&z2),
        TensorContractSpec::with_default_output_order(&[], &[]),
    );

    assert_rule_mismatch(error);
}

#[test]
fn fusion_contract_rejects_same_type_rule_with_distinct_identity() {
    let first = SymbolOnlyRule::new(1.0);
    let second = SymbolOnlyRule::new(-1.0);
    let error = scalar_contract_error(
        &second,
        bound_scalar_space(&first),
        bound_scalar_space(&first),
        bound_scalar_space(&first),
        TensorContractSpec::with_default_output_order(&[], &[]),
    );

    assert_rule_mismatch(error);
}

#[test]
fn fusion_contract_bound_same_rule_matches_on_direct_and_fallback_routes() {
    let rule = Z2FusionRule;
    let space = || bind(&rule, &bound_scalar_space(&rule)).unwrap();
    let (dst_space, lhs, rhs) = (space(), space(), space());
    for lhs_operand in [
        FusionOperand::direct(lhs.space()),
        FusionOperand::adjoint(lhs.space()),
    ] {
        let mut dst = [7.0];

        TensorContractFusionExecutionContext::<f64, _>::default()
            .tensorcontract_planned_into(
                &dst_space,
                &mut dst,
                (lhs_operand, &[2.0]),
                (FusionOperand::direct(rhs.space()), &[5.0]),
                &[],
                &[],
                &[],
                2.0,
                ContractDestinationInit::Axpby(3.0),
            )
            .unwrap();

        assert_eq!(dst, [41.0]);
    }
}

#[test]
fn dynamic_trace_rejects_same_type_rule_when_only_symbols_differ() {
    let first = SymbolOnlyRule::new(1.0);
    let second = SymbolOnlyRule::new(-1.0);
    let space = DynamicFusionMapSpace::from_final_homspace(
        &first,
        FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
    )
    .unwrap();

    // What: the wrong provider is rejected at the one raw-to-bound ingress;
    // public trace APIs cannot accept an independent provider afterwards.
    let error =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(space, Arc::new(second)).unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}
