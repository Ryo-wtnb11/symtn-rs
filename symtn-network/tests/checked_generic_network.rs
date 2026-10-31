#![cfg(feature = "cgc-gen")]

use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::LieGroupFusionRule;
use symtn::sector::SectorId;
use symtn::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorVec, TypedSectorAdmission,
};
use symtn::typed::CheckedGenericStructureError;
use symtn::typed::{
    BlockFusionTrees, CheckedGenericPlanError, GenericTensorError, GradedSpace, TensorMap,
};
use symtn::typed::{Complex64, ContractSpec, Runtime, TensorScalar};
#[cfg(feature = "opt-path")]
use symtn_network::Optimizer;
use symtn_network::{
    configure_plan_cache, plan_cache_config, plan_cache_stats, slice_plan_for, DegeneracyRange,
    DenseCostModel, DenseTensorInfo, GreedyDenseOptimizer, LabelOrderDenseOptimizer, Network,
    NetworkExecutionWorkspace, NetworkIR, PlanCacheConfig, PlannedNetwork, SectorSlice, SlicedPlan,
    SymmetricSliceExecutionError, SymmetricSlicePlan, SymmetricSliceSpec, SymmetricSlicedPlan,
    TemporaryLabel, TensorId,
};

use crate::network_support;
use network_support::{net, op};

trait OracleScalar: TensorScalar + Copy + Debug {
    fn value(marker: usize) -> Self;
    fn distance(self, other: Self) -> f64;
}

impl OracleScalar for f64 {
    fn value(marker: usize) -> Self {
        marker as f64 / 17.0
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).abs()
    }
}

impl OracleScalar for Complex64 {
    fn value(marker: usize) -> Self {
        Self::new(marker as f64 / 17.0, -(marker as f64) / 23.0)
    }

    fn distance(self, other: Self) -> f64 {
        (self - other).norm()
    }
}

fn marker(trees: &BlockFusionTrees<Vec<i64>>, indices: &[usize]) -> usize {
    trees
        .codomain_vertices()
        .iter()
        .chain(trees.domain_vertices())
        .enumerate()
        .map(|(index, vertex)| (index + 1) * 100 * vertex.get())
        .chain(
            indices
                .iter()
                .enumerate()
                .map(|(index, value)| (index + 1) * value),
        )
        .sum()
}

fn assert_same<D: OracleScalar>(
    actual: &TensorMap<LieGroupFusionRule, D>,
    expected: &TensorMap<LieGroupFusionRule, D>,
) {
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    for index in 0..actual.subblock_count() {
        assert_eq!(
            actual.subblock_fusion_trees(index).unwrap(),
            expected.subblock_fusion_trees(index).unwrap()
        );
        assert_eq!(
            actual.subblock(index).unwrap(),
            expected.subblock(index).unwrap()
        );
    }
    assert_eq!(
        actual.dense_data().unwrap().len(),
        expected.dense_data().unwrap().len()
    );
    for (index, (&lhs, &rhs)) in actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .enumerate()
    {
        assert!(
            lhs.distance(rhs) <= 1.0e-10 * (1.0 + lhs.distance(D::value(0))),
            "payload {index} differs: {lhs:?} vs {rhs:?}"
        );
    }
}

fn authority_provider<D>(
    planned: &PlannedNetwork,
    tensors: &[&TensorMap<LieGroupFusionRule, D>],
) -> *const LieGroupFusionRule {
    let mut authorities = tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| (TensorId::new(index), tensor.provider() as *const _))
        .collect::<HashMap<_, _>>();
    for step in planned.plan().steps() {
        let authority = authorities[&step.lhs()];
        authorities.remove(&step.lhs());
        authorities.remove(&step.rhs());
        authorities.insert(step.result(), authority);
    }
    authorities[&planned.plan().steps().last().unwrap().result()]
}

fn assert_sun_network<D: OracleScalar + Send + Sync + 'static>(
    engine: &mut dyn Engine,
    n: usize,
    label: Vec<i64>,
) {
    let runtime = Runtime::builder().build().unwrap();
    let lhs_provider = Arc::new(symtn_core::testing::su_lie_rule(n));
    let rhs_provider = Arc::new(symtn_core::testing::su_lie_rule(n));
    let tail_provider = Arc::new(symtn_core::testing::su_lie_rule(n));
    let lhs_leg = GradedSpace::try_new(Arc::clone(&lhs_provider), [(label.clone(), 1)]).unwrap();
    let rhs_leg = GradedSpace::try_new(Arc::clone(&rhs_provider), [(label.clone(), 1)]).unwrap();
    let tail_leg = GradedSpace::try_new(Arc::clone(&tail_provider), [(label.clone(), 1)]).unwrap();
    let lhs: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&lhs_leg, &lhs_leg],
        [&lhs_leg],
        |trees, indices| D::value(10_000 + marker(trees, indices)),
    )
    .unwrap();
    assert!((0..lhs.subblock_count()).any(|index| {
        lhs.subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)
    }));
    let middle: TensorMap<_, D> =
        TensorMap::from_subblock_fn(engine, &runtime, [&rhs_leg], [&rhs_leg], |_, _| {
            D::value(17)
        })
        .unwrap();
    let tail: TensorMap<_, D> =
        TensorMap::from_subblock_fn(engine, &runtime, [&tail_leg], [&tail_leg], |_, _| {
            D::value(17)
        })
        .unwrap();
    assert!(!std::ptr::eq(lhs.provider(), middle.provider()));

    let chain = lhs
        .contract(
            engine,
            &middle,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap()
        .contract(
            engine,
            &tail,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    let expected = chain.permute(engine, &[1, 0], &[2]).unwrap();
    let network = Network::new(
        vec![vec!["a", "b", "c"], vec!["c", "d"], vec!["d", "e"]],
        vec![false; 3],
        vec![2, 1, 1],
        vec!["b", "a", "e"],
        Some(2),
    )
    .unwrap();
    let tensors = [&lhs, &middle, &tail];
    let explicit = network
        .plan(
            &tensors,
            &LabelOrderDenseOptimizer::new(vec!["c".into(), "d".into()]),
        )
        .unwrap();
    let greedy = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for planned in [&explicit, &greedy] {
        let authority = authority_provider(planned, &tensors);
        for _ in 0..2 {
            let actual = planned.execute(engine, &tensors, &mut workspace).unwrap();
            assert_eq!(actual.provider() as *const _, authority);
            assert_same(&actual, &expected);
        }
    }
    let mut separate_a = NetworkExecutionWorkspace::default();
    let mut separate_b = NetworkExecutionWorkspace::default();
    assert_same(
        &explicit.execute(engine, &tensors, &mut separate_a).unwrap(),
        &expected,
    );
    assert_same(
        &explicit.execute(engine, &tensors, &mut separate_b).unwrap(),
        &expected,
    );

    let drift_lhs_provider = Arc::new(symtn_core::testing::su_lie_rule(n));
    let drift_middle_provider = Arc::new(symtn_core::testing::su_lie_rule(n));
    let drift_tail_provider = Arc::new(symtn_core::testing::su_lie_rule(n));
    let drift_lhs_leg =
        GradedSpace::try_new(Arc::clone(&drift_lhs_provider), [(label.clone(), 1)]).unwrap();
    let drift_middle_leg =
        GradedSpace::try_new(Arc::clone(&drift_middle_provider), [(label.clone(), 1)]).unwrap();
    let drift_tail_leg =
        GradedSpace::try_new(Arc::clone(&drift_tail_provider), [(label, 1)]).unwrap();
    let drift_lhs: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&drift_lhs_leg, &drift_lhs_leg],
        [&drift_lhs_leg],
        |trees, indices| D::value(10_000 + marker(trees, indices)),
    )
    .unwrap();
    let drift_middle: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&drift_middle_leg],
        [&drift_middle_leg],
        |_, _| D::value(17),
    )
    .unwrap();
    let drift_tail: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&drift_tail_leg],
        [&drift_tail_leg],
        |_, _| D::value(17),
    )
    .unwrap();
    let drift_refs = [&drift_lhs, &drift_middle, &drift_tail];
    let drift = explicit
        .execute(engine, &drift_refs, &mut separate_a)
        .unwrap();
    assert_eq!(
        drift.provider() as *const _,
        authority_provider(&explicit, &drift_refs)
    );
    assert_same(&drift, &expected);

    let cached_first = net(
        &[
            op(&["a", "b"], &["c"]),
            op(&["c"], &["d"]),
            op(&["d"], &["e"]),
        ],
        &["b", "a"],
        &["e"],
    )
    .contract(engine, &[&lhs, &middle, &tail])
    .unwrap();
    let cached_replay = net(
        &[
            op(&["a", "b"], &["c"]),
            op(&["c"], &["d"]),
            op(&["d"], &["e"]),
        ],
        &["b", "a"],
        &["e"],
    )
    .contract(engine, &[&lhs, &middle, &tail])
    .unwrap();
    assert_eq!(
        cached_first.provider() as *const _,
        authority_provider(&greedy, &tensors)
    );
    assert_same(&cached_first, &expected);
    assert_same(&cached_replay, &expected);
    // An output that moves `a` across the split is the last step's own
    // ContractSpec; it equals the chain followed by that permute.
    let moved = net(
        &[
            op(&["a", "b"], &["c"]),
            op(&["c"], &["d"]),
            op(&["d"], &["e"]),
        ],
        &["b"],
        &["a", "e"],
    )
    .contract(engine, &[&lhs, &middle, &tail])
    .unwrap();
    assert_same(&moved, &chain.permute(engine, &[1], &[0, 2]).unwrap());

    let planned = Arc::new(greedy);
    let operands = [Arc::new(lhs), Arc::new(middle), Arc::new(tail)];
    let workers = (0..2)
        .map(|_| {
            let planned = Arc::clone(&planned);
            let operands = operands.clone();
            std::thread::spawn(move || {
                let engine = &mut symtn::typed::CpuEngine::serial();
                planned
                    .execute(
                        engine,
                        &[&operands[0], &operands[1], &operands[2]],
                        &mut Default::default(),
                    )
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        assert_same(&worker.join().unwrap(), &expected);
    }
}

#[test]
fn sun_checked_generic_network_matches_manual_explicit_greedy_cached_and_replay() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: provider-neutral SU(N) conformance over μ=2 keys, both payload
    // dtypes, three operands, nontrivial final order, replay, and concurrency.
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_network::<f64>(engine, n, label.clone());
        assert_sun_network::<Complex64>(engine, n, label);
    }
}

#[test]
fn sun_checked_generic_mixed_slice_preserves_outer_multiplicity_keys() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(symtn_core::testing::su_lie_rule(3));
    let label = vec![1, 1];
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 2)]).unwrap();
    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, indices| {
            Complex64::value(20_000 + marker(trees, indices))
        })
        .unwrap();
    assert!((0..lhs.subblock_count()).any(|index| {
        lhs.subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)
    }));
    let rhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, indices| {
            Complex64::value(30_000 + marker(trees, indices))
        })
        .unwrap();
    let inputs = vec![
        ["a", "b", "x"].map(TemporaryLabel::from).to_vec(),
        ["x", "d"].map(TemporaryLabel::from).to_vec(),
    ];
    let output = ["a", "b", "d"].map(TemporaryLabel::from).to_vec();
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![2, 1],
        output.clone(),
        Some(2),
    )
    .unwrap();
    let tensors = [&lhs, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let expected = planned
        .execute(engine, &tensors, &mut Default::default())
        .unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        slice_plan_for(
            &ir,
            planned.plan(),
            &cost,
            &["a", "x"].map(TemporaryLabel::from),
        ),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    let (actual, stats) = network
        .execute_symmetric_sliced(engine, &tensors, sliced, usize::MAX)
        .unwrap();
    assert!(stats.peak_total_bytes() > 0);
    assert_eq!(actual.provider() as *const _, provider.as_ref() as *const _);
    assert_same(&actual, &expected);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InjectedError {
    Decode,
    Encode,
    Dual,
    Symbol,
    Provider,
}

impl fmt::Display for InjectedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "injected {self:?} failure")
    }
}

impl std::error::Error for InjectedError {}

struct InjectedGeneric {
    inner: LieGroupFusionRule,
    fail_encode: AtomicBool,
    fail_decode_at: AtomicUsize,
    decode_calls: AtomicUsize,
    fail_dual: AtomicBool,
    dual_calls: AtomicUsize,
    fail_symbol_at: AtomicUsize,
    symbol_calls: AtomicUsize,
}

impl InjectedGeneric {
    fn new() -> Self {
        Self {
            inner: symtn_core::testing::su_lie_rule(3),
            fail_encode: AtomicBool::new(false),
            fail_decode_at: AtomicUsize::new(0),
            decode_calls: AtomicUsize::new(0),
            fail_dual: AtomicBool::new(false),
            dual_calls: AtomicUsize::new(0),
            fail_symbol_at: AtomicUsize::new(0),
            symbol_calls: AtomicUsize::new(0),
        }
    }

    fn reset_symbols(&self) {
        self.fail_symbol_at.store(0, Ordering::SeqCst);
        self.symbol_calls.store(0, Ordering::SeqCst);
    }

    fn arm_symbol(&self, ordinal: usize) {
        self.symbol_calls.store(0, Ordering::SeqCst);
        self.fail_symbol_at.store(ordinal, Ordering::SeqCst);
    }

    fn symbol(&self) -> Result<(), InjectedError> {
        let ordinal = self.symbol_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_symbol_at.load(Ordering::SeqCst) == ordinal {
            Err(InjectedError::Symbol)
        } else {
            Ok(())
        }
    }

    fn reset_decodes(&self) {
        self.fail_decode_at.store(0, Ordering::SeqCst);
        self.decode_calls.store(0, Ordering::SeqCst);
    }

    fn arm_decode(&self, ordinal: usize) {
        self.decode_calls.store(0, Ordering::SeqCst);
        self.fail_decode_at.store(ordinal, Ordering::SeqCst);
    }
}

impl symtn_core::testing::Provider for InjectedGeneric {}

impl CheckedGenericFusion for InjectedGeneric {
    type Error = InjectedError;

    fn rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(&self.inner)
    }

    fn fusion_style(&self) -> FusionStyleKind {
        CheckedGenericFusion::fusion_style(&self.inner)
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        CheckedGenericFusion::braiding_style(&self.inner)
    }

    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.inner)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.dual_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_dual.load(Ordering::SeqCst) {
            Err(InjectedError::Dual)
        } else {
            CheckedGenericFusion::try_dual(&self.inner, sector).map_err(|_| InjectedError::Provider)
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_fusion_channels(&self.inner, left, right)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_fusion_channels_in_table(&self.inner, left, right)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        self.symbol()?;
        CheckedGenericFusion::try_nsymbol(&self.inner, left, right, coupled)
            .map_err(|_| InjectedError::Provider)
    }
}

#[cfg(feature = "opt-path")]
#[derive(Clone, Copy)]
enum PlanningFailure {
    Dual,
    Dimension,
}

#[cfg(feature = "opt-path")]
fn assert_optimizer_does_not_retry_provider_failure(
    optimizer: Optimizer,
    failure: PlanningFailure,
) {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            optimizer,
            ..PlanCacheConfig::default()
        },
    );
    let provider = Arc::new(InjectedGeneric::new());
    let tensors = injected_chain(engine, &runtime, &provider);
    provider.dual_calls.store(0, Ordering::SeqCst);
    match failure {
        PlanningFailure::Dual => provider.fail_dual.store(true, Ordering::SeqCst),
        PlanningFailure::Dimension => provider.arm_symbol(1),
    }
    let first = &tensors[0];
    let second = &tensors[1];
    let third = &tensors[2];
    let fourth = &tensors[3];
    let error = net(
        &[
            op(&["a", "b"], &["c"]),
            op(&["c"], &["d"]),
            op(&["d"], &["e"]),
            op(&["e"], &["f"]),
        ],
        &["a", "b"],
        &["f"],
    )
    .contract(engine, &[first, second, third, fourth])
    .unwrap_err();
    match failure {
        PlanningFailure::Dual => {
            assert!(matches!(
                error,
                GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                    InjectedError::Dual
                ))
            ));
            assert_eq!(provider.dual_calls.load(Ordering::SeqCst), 1);
        }
        PlanningFailure::Dimension => {
            assert!(matches!(
                error,
                GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                    InjectedError::Symbol
                ))
            ));
            assert_eq!(provider.symbol_calls.load(Ordering::SeqCst), 1);
        }
    }
    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 0);
    assert_eq!(stats.topology_materializations, 0);
    assert_eq!(stats.workspaces_created, 0);
}

#[cfg(feature = "opt-path")]
#[test]
fn checked_generic_optimizer_fallback_never_retries_provider_failures() {
    let _serial = crate::counting_alloc::serial();
    for optimizer in [Optimizer::DynamicProgramming, Optimizer::AutoHq] {
        for failure in [PlanningFailure::Dual, PlanningFailure::Dimension] {
            assert_optimizer_does_not_retry_provider_failure(optimizer.clone(), failure);
        }
    }
}

impl CheckedGenericRigidSymbols for InjectedGeneric {
    type Scalar = f64;

    fn try_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_sqrt_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_inv_sqrt_dim_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_frobenius_schur_phase_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_f_symbol_generic(&self.inner, a, b, c, d, e, f)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.symbol()?;
        CheckedGenericRigidSymbols::try_r_symbol_generic(&self.inner, a, b, c)
            .map_err(|_| InjectedError::Provider)
    }
}

impl CheckedGenericPivotal for InjectedGeneric {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol()?;
        CheckedGenericPivotal::try_twist_scalar(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }
}

impl TypedSectorAdmission for InjectedGeneric {
    type Sector = Vec<i64>;
    type Error = InjectedError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        if self.fail_encode.load(Ordering::SeqCst) {
            return Err(InjectedError::Encode);
        }
        TypedSectorAdmission::try_encode_label(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        let ordinal = self.decode_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.fail_decode_at.load(Ordering::SeqCst) == ordinal {
            return Err(InjectedError::Decode);
        }
        TypedSectorAdmission::try_decode_label(&self.inner, sector)
            .map_err(|_| InjectedError::Provider)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.try_dual(sector)
    }
}

fn injected_chain(
    engine: &mut dyn Engine,
    runtime: &Runtime,
    provider: &Arc<InjectedGeneric>,
) -> Vec<TensorMap<InjectedGeneric, f64>> {
    let leg = GradedSpace::try_new(Arc::clone(provider), [(vec![1, 1], 1)]).unwrap();
    vec![
        TensorMap::from_subblock_fn(engine, runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap(),
        TensorMap::from_subblock_fn(engine, runtime, [&leg], [&leg], |_, _| 2.0).unwrap(),
        TensorMap::from_subblock_fn(engine, runtime, [&leg], [&leg], |_, _| 3.0).unwrap(),
        TensorMap::from_subblock_fn(engine, runtime, [&leg], [&leg], |_, _| 4.0).unwrap(),
    ]
}

fn injected_network(operands: usize, permute_output: bool) -> Network {
    let all_inputs = vec![
        vec!["a", "b", "c"],
        vec!["c", "d"],
        vec!["d", "e"],
        vec!["e", "f"],
    ];
    let tail = ["c", "d", "e", "f"][operands - 1];
    Network::new(
        all_inputs.into_iter().take(operands),
        vec![false; operands],
        std::iter::once(2).chain(std::iter::repeat_n(1, operands - 1)),
        if permute_output {
            vec!["b", "a", tail]
        } else {
            vec!["a", "b", tail]
        },
        Some(2),
    )
    .unwrap()
}

fn assert_injected_recovery(
    engine: &mut dyn Engine,
    provider: &InjectedGeneric,
    planned: &PlannedNetwork,
    tensors: &[&TensorMap<InjectedGeneric, f64>],
    ordinal: usize,
) {
    let mut workspace = NetworkExecutionWorkspace::default();
    // The isolated subprocess owns this global reset. Discard layouts built by
    // the fixture so `ordinal` names a query of the execution under test.
    symtn::cache::clear();
    provider.arm_symbol(ordinal);
    assert!(matches!(
        planned.execute(engine, tensors, &mut workspace),
        Err(GenericTensorError::Plan(_))
    ));
    provider.reset_symbols();
    let recovered = planned.execute(engine, tensors, &mut workspace).unwrap();
    provider.reset_symbols();
    let expected = planned
        .execute(engine, tensors, &mut Default::default())
        .unwrap();
    assert_eq!(recovered.subblock_count(), expected.subblock_count());
    for (&actual, &want) in recovered
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
    {
        assert!((actual - want).abs() <= 1.0e-12 * (1.0 + want.abs()));
    }
}

fn injected_plan_case(
    engine: &mut dyn Engine,
    operands: usize,
    permute_output: bool,
) -> (
    Arc<InjectedGeneric>,
    Vec<TensorMap<InjectedGeneric, f64>>,
    PlannedNetwork,
) {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let tensors = injected_chain(engine, &runtime, &provider);
    let refs = tensors.iter().take(operands).collect::<Vec<_>>();
    let planned = injected_network(operands, permute_output)
        .plan(
            &refs,
            &LabelOrderDenseOptimizer::new(
                ["c", "d", "e"][..operands - 1]
                    .iter()
                    .map(|&label| label.into())
                    .collect(),
            ),
        )
        .unwrap();
    (provider, tensors, planned)
}

fn cold_query_count(engine: &mut dyn Engine, operands: usize, permute_output: bool) -> usize {
    let (provider, tensors, planned) = injected_plan_case(engine, operands, permute_output);
    let refs = tensors.iter().take(operands).collect::<Vec<_>>();
    // Each spy has the same semantic identity, so measure after discarding a
    // layout an earlier fixture may legitimately have published.
    symtn::cache::clear();
    provider.reset_symbols();
    planned
        .execute(engine, &refs, &mut Default::default())
        .unwrap();
    provider.symbol_calls.load(Ordering::SeqCst)
}

#[test]
fn checked_generic_failures_stay_typed_and_workspace_recovers_at_every_step() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_NETWORK_FAILURES_ISOLATED";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "checked_generic_network::checked_generic_failures_stay_typed_and_workspace_recovers_at_every_step",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success()
                && String::from_utf8_lossy(&output.stdout)
                    .contains("test result: ok. 1 passed; 0 failed;"),
            "isolated checked-Generic network failure test failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());

    provider.fail_encode.store(true, Ordering::SeqCst);
    assert!(matches!(
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]),
        Err(GenericTensorError::Structure(_))
    ));
    provider.fail_encode.store(false, Ordering::SeqCst);

    let tensors = injected_chain(engine, &runtime, &provider);
    let refs = tensors.iter().collect::<Vec<_>>();
    provider.fail_dual.store(true, Ordering::SeqCst);
    assert!(matches!(
        injected_network(4, true).plan(&refs, &GreedyDenseOptimizer),
        Err(GenericTensorError::Structure(_))
    ));
    provider.fail_dual.store(false, Ordering::SeqCst);

    let first_end = cold_query_count(engine, 2, false);
    let middle_end = cold_query_count(engine, 3, false);
    let final_end = cold_query_count(engine, 4, false);
    let permutation_end = cold_query_count(engine, 4, true);
    assert!([first_end, middle_end, final_end, permutation_end]
        .into_iter()
        .all(|queries| queries > 0));

    for (operands, permute, ordinal) in [
        (2, false, 1),
        (3, false, middle_end),
        (4, false, final_end),
        (4, true, permutation_end),
    ] {
        let (case_provider, case_tensors, case_plan) =
            injected_plan_case(engine, operands, permute);
        let case_refs = case_tensors.iter().take(operands).collect::<Vec<_>>();
        assert_injected_recovery(engine, &case_provider, &case_plan, &case_refs, ordinal);
    }
}

#[test]
fn checked_generic_sliced_late_provider_failure_is_typed_and_recovers() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            (1 + ij[0] + 3 * ij[1]) as f64
        })
        .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            (2 + 2 * ij[0] + ij[1]) as f64
        })
        .unwrap();
    let inputs = vec![
        ["a", "x"].map(TemporaryLabel::from).to_vec(),
        ["x", "c"].map(TemporaryLabel::from).to_vec(),
    ];
    let output = ["a", "c"].map(TemporaryLabel::from).to_vec();
    let ir = NetworkIR::from_labels(inputs.clone(), output.clone()).unwrap();
    let network = Network::new(inputs, vec![false; 2], vec![1; 2], output, Some(1)).unwrap();
    let tensors = [&lhs, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        slice_plan_for(&ir, planned.plan(), &cost, &[TemporaryLabel::from("a")]),
    );
    let multi = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();
    let index = &multi.slices().indices()[0];
    assert_eq!(index.output_position(), Some(0));
    assert_eq!(index.pieces().len(), 2);
    let sector = index.pieces()[0].sector();
    assert!(index.pieces().iter().all(|piece| piece.sector() == sector));
    let single_slice = SymmetricSlicePlan::try_new(
        &ir,
        multi.slices().rule_identity().clone(),
        vec![SymmetricSliceSpec::new(
            index.label().clone(),
            index.authority(),
            index.authority_leg().clone(),
            vec![SectorSlice::new(
                sector,
                DegeneracyRange::new(0, 2).unwrap(),
            )],
        )],
    )
    .unwrap();
    let single = SymmetricSlicedPlan::new(multi.plan().clone(), single_slice);

    // Warm provider-owned catalogs, then use the one-job output execution to
    // locate the first checked decode after a completed output rectangle. The
    // two-rectangle run has the same bind/compile/first-job sequence and
    // strictly more decode calls.
    network
        .execute_symmetric_sliced(engine, &tensors, multi.clone(), usize::MAX)
        .unwrap();
    provider.reset_decodes();
    network
        .execute_symmetric_sliced(engine, &tensors, single, usize::MAX)
        .unwrap();
    let first_rectangle_end = provider.decode_calls.load(Ordering::SeqCst);
    provider.reset_decodes();
    let (expected, _) = network
        .execute_symmetric_sliced(engine, &tensors, multi.clone(), usize::MAX)
        .unwrap();
    let all_jobs_end = provider.decode_calls.load(Ordering::SeqCst);
    assert!(first_rectangle_end > 0);
    assert!(all_jobs_end > first_rectangle_end);

    provider.arm_decode(first_rectangle_end + 1);
    let error = network
        .execute_symmetric_sliced(engine, &tensors, multi.clone(), usize::MAX)
        .unwrap_err();
    assert!(matches!(
        error,
        SymmetricSliceExecutionError::Tensor(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(InjectedError::Decode)
        ))
    ));
    assert_eq!(
        provider.decode_calls.load(Ordering::SeqCst),
        first_rectangle_end + 1
    );

    provider.reset_decodes();
    let (recovered, _) = network
        .execute_symmetric_sliced(engine, &tensors, multi, usize::MAX)
        .unwrap();
    assert_eq!(
        recovered.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
}

#[test]
fn checked_generic_trace_provider_failure_stays_typed_and_recovers() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a provider error inside `trace_pairs`, the first step of the
    // two-step trace-then-network form, returns the typed provider error,
    // and the same trace succeeds once the provider recovers.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
    provider.arm_symbol(1);
    let error = tensor.trace_pairs(engine, &[(0, 1)]).unwrap_err();
    assert!(
        matches!(
            error,
            GenericTensorError::Plan(CheckedGenericPlanError::Provider(InjectedError::Symbol))
        ),
        "{error:?}"
    );

    provider.reset_symbols();
    let traced = tensor.trace_pairs(engine, &[(0, 1)]).unwrap();
    assert_eq!(traced.provider() as *const _, provider.as_ref() as *const _);
}

#[test]
fn checked_generic_scalar_empty_outer_product_and_single_permute_follow_ordinary_ops() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [], |_, _| 2.0).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [], [&leg], |_, _| 5.0).unwrap();
    let scalar = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[0],
                rhs: &[0],
                codomain: &[],
                domain: &[],
            },
        )
        .unwrap();
    let scalar_plan = Network::new([Vec::<&str>::new()], [false], [0], [], Some(0))
        .unwrap()
        .plan(&[&scalar], &GreedyDenseOptimizer)
        .unwrap();
    assert_eq!(
        scalar_plan
            .execute(engine, &[&scalar], &mut Default::default())
            .unwrap()
            .dense_data()
            .unwrap(),
        scalar.dense_data().unwrap()
    );
    let outer = Network::new(
        vec![vec!["a"], vec!["b"]],
        vec![false; 2],
        vec![1, 0],
        vec!["a", "b"],
        Some(1),
    )
    .unwrap()
    .plan(&[&lhs, &rhs], &GreedyDenseOptimizer)
    .unwrap()
    .execute(engine, &[&lhs, &rhs], &mut Default::default())
    .unwrap();
    let expected_outer = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[],
                rhs: &[],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(
        outer.dense_data().unwrap(),
        expected_outer.dense_data().unwrap()
    );

    let rank_three: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let permuted = Network::new(
        vec![vec!["a", "b", "c"]],
        vec![false],
        vec![2],
        vec!["b", "a", "c"],
        Some(2),
    )
    .unwrap()
    .plan(&[&rank_three], &GreedyDenseOptimizer)
    .unwrap()
    .execute(engine, &[&rank_three], &mut Default::default())
    .unwrap();
    assert_eq!(
        permuted.dense_data().unwrap(),
        rank_three
            .permute(engine, &[1, 0], &[2])
            .unwrap()
            .dense_data()
            .unwrap()
    );

    let empty = GradedSpace::try_new(
        Arc::clone(&provider),
        std::iter::empty::<(Vec<i64>, usize)>(),
    )
    .unwrap();
    let zero: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&empty], []).unwrap();
    let zero_outer = Network::new(
        vec![vec!["a"], vec!["b"]],
        vec![false; 2],
        vec![1; 2],
        vec!["a", "b"],
        Some(1),
    )
    .unwrap()
    .plan(&[&zero, &zero], &GreedyDenseOptimizer)
    .unwrap()
    .execute(engine, &[&zero, &zero], &mut Default::default())
    .unwrap();
    assert!(zero_outer.dense_data().unwrap().is_empty());
}

#[test]
fn checked_generic_cache_modes_dtype_pools_and_lazy_operands_match_direct_authority() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(InjectedGeneric::new());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let a64: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let b64: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 3.0).unwrap();
    let ac: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            Complex64::new(2.0, 1.0)
        })
        .unwrap();
    let bc: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            Complex64::new(3.0, -1.0)
        })
        .unwrap();
    let first = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[&a64, &b64])
        .unwrap();
    let second = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[&a64, &b64])
        .unwrap();
    let complex = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[&ac, &bc])
        .unwrap();
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());
    assert_eq!(
        complex.dense_data().unwrap(),
        ac.contract(
            engine,
            &bc,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .unwrap()
        .dense_data()
        .unwrap()
    );
    let stats = plan_cache_stats(&runtime);
    assert!(stats.hits >= 2);
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.workspaces_created, 2);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            enabled: false,
            ..plan_cache_config(&runtime)
        },
    );
    let uncached = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[&a64, &b64])
        .unwrap();
    assert_eq!(uncached.dense_data().unwrap(), first.dense_data().unwrap());

    let lazy = a64.adjoint(engine).unwrap();
    let network = Network::new(
        vec![vec!["i", "j"], vec!["j", "k"]],
        vec![false; 2],
        vec![1; 2],
        vec!["i", "k"],
        Some(1),
    )
    .unwrap();
    let planned = network.plan(&[&lazy, &b64], &GreedyDenseOptimizer).unwrap();
    let direct = lazy.contract(
        engine,
        &b64,
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
    );
    let replay = planned.execute(engine, &[&lazy, &b64], &mut Default::default());
    // A checked lazy adjoint is a storage-conjugate operand (#1865): the
    // replay and a conjugated network leg agree with the direct contraction
    // and with the materialized adjoint, and the lazy input stays lazy.
    let direct = direct.unwrap();
    let expected = lazy
        .materialize(engine)
        .unwrap()
        .contract(
            engine,
            &b64,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(direct.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(
        replay.unwrap().dense_data().unwrap(),
        direct.dense_data().unwrap()
    );
    assert!(matches!(
        symtn::typed::__network::network_reuse_class(&lazy, false),
        symtn::typed::__network::NetworkReuseClass::LazyAdjoint
    ));

    let conjugated = Network::new(
        vec![vec!["j", "i"], vec!["j", "k"]],
        vec![true, false],
        vec![1; 2],
        vec!["i", "k"],
        Some(1),
    )
    .unwrap()
    .plan(&[&a64, &b64], &GreedyDenseOptimizer)
    .unwrap()
    .execute(engine, &[&a64, &b64], &mut Default::default());
    assert_eq!(
        conjugated.unwrap().dense_data().unwrap(),
        direct.dense_data().unwrap()
    );
}

#[test]
fn checked_warm_replay_matches_eager_chain_and_multiplicity_free_su2() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a checked Generic replay that overwrites its retained step
    // destinations (`contract_into`) gives the eager chain's values and the
    // multiplicity-free SU(2) replay of the same spin-½ blocks.
    use symtn::sector::{SU2FusionRule, SU2Irrep};

    fn fill(seed: usize) -> impl Fn(&[usize]) -> f64 {
        move |indices| (1 + seed + 7 * indices[0] + 3 * indices[1]) as f64 / 16.0
    }
    let runtime = Runtime::builder().build().unwrap();
    let network = net(
        &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["a"], &["d"])],
        &["c"],
        &["d"],
    );
    let first = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[1],
        domain: &[0],
    };
    let second = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    let degeneracies = [(5, 4), (4, 6), (5, 6)];

    let checked = Arc::new(symtn_core::testing::su_lie_rule(2));
    let checked_space = |dim| GradedSpace::try_new(Arc::clone(&checked), [(vec![1], dim)]).unwrap();
    let checked_tensors: Vec<TensorMap<LieGroupFusionRule, f64>> = degeneracies
        .iter()
        .enumerate()
        .map(|(seed, &(rows, cols))| {
            let value = fill(seed);
            TensorMap::from_subblock_fn(
                engine,
                &runtime,
                // The third operand's `a` leg pairs with the first one's.
                [&if seed == 2 {
                    checked_space(rows).try_dual().unwrap()
                } else {
                    checked_space(rows)
                }],
                [&checked_space(cols)],
                |_, indices| value(indices),
            )
            .unwrap()
        })
        .collect();
    let checked_refs = [
        &checked_tensors[0],
        &checked_tensors[1],
        &checked_tensors[2],
    ];
    let expected = checked_tensors[0]
        .contract(engine, &checked_tensors[1], &first)
        .unwrap()
        .contract(engine, &checked_tensors[2], &second)
        .unwrap();
    let checked_plan = network.plan(&checked_refs, &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    let cold = checked_plan
        .execute(engine, &checked_refs, &mut workspace)
        .unwrap();
    let warm = checked_plan
        .execute(engine, &checked_refs, &mut workspace)
        .unwrap();
    assert_same(&cold, &expected);
    assert_same(&warm, &expected);

    let mf = Arc::new(SU2FusionRule);
    let mf_space =
        |dim| GradedSpace::try_new(Arc::clone(&mf), [(SU2Irrep::from_twice_spin(1), dim)]).unwrap();
    let mf_tensors: Vec<TensorMap<SU2FusionRule, f64>> = degeneracies
        .iter()
        .enumerate()
        .map(|(seed, &(rows, cols))| {
            let value = fill(seed);
            TensorMap::from_subblock_fn(
                engine,
                &runtime,
                // The third operand's `a` leg pairs with the first one's.
                [&if seed == 2 {
                    mf_space(rows).try_dual().unwrap()
                } else {
                    mf_space(rows)
                }],
                [&mf_space(cols)],
                |_, indices| value(indices),
            )
            .unwrap()
        })
        .collect();
    let mf_refs = [&mf_tensors[0], &mf_tensors[1], &mf_tensors[2]];
    let mf_plan = network.plan(&mf_refs, &GreedyDenseOptimizer).unwrap();
    let mut mf_workspace = NetworkExecutionWorkspace::default();
    drop(
        mf_plan
            .execute(engine, &mf_refs, &mut mf_workspace)
            .unwrap(),
    );
    let mf_warm = mf_plan
        .execute(engine, &mf_refs, &mut mf_workspace)
        .unwrap();
    let (checked_data, mf_data) = (warm.dense_data().unwrap(), mf_warm.dense_data().unwrap());
    assert_eq!(checked_data.len(), mf_data.len());
    for (index, (&lhs, &rhs)) in checked_data.iter().zip(mf_data).enumerate() {
        assert!(
            (lhs - rhs).abs() <= 1.0e-10 * (1.0 + lhs.abs()),
            "payload {index}: checked {lhs} vs multiplicity-free {rhs}"
        );
    }
}

#[test]
fn checked_destination_reuse_queries_no_more_provider_symbols_than_a_fresh_replay() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: overwriting the kept step destinations (`contract_into`) stages
    // the same contractions as the returning replay, so a replay through a
    // kept workspace asks the provider for no more symbols than one through a
    // fresh workspace, and gives the same values.
    const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_NETWORK_REUSE_LEDGER_ISOLATED";
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "checked_generic_network::checked_destination_reuse_queries_no_more_provider_symbols_than_a_fresh_replay",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success()
                && String::from_utf8_lossy(&output.stdout)
                    .contains("test result: ok. 1 passed; 0 failed;"),
            "isolated checked-Generic reuse ledger test failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    let (provider, tensors, planned) = injected_plan_case(engine, 4, true);
    let refs = tensors.iter().collect::<Vec<_>>();
    let mut kept = NetworkExecutionWorkspace::default();
    drop(planned.execute(engine, &refs, &mut kept).unwrap());
    // Both measured replays start from empty structure caches, so each one
    // stages its contractions from provider queries.
    symtn::cache::clear();
    provider.reset_symbols();
    let fresh = planned
        .execute(engine, &refs, &mut Default::default())
        .unwrap();
    let fresh_calls = provider.symbol_calls.load(Ordering::SeqCst);
    symtn::cache::clear();
    provider.reset_symbols();
    let reused = planned.execute(engine, &refs, &mut kept).unwrap();
    let reused_calls = provider.symbol_calls.load(Ordering::SeqCst);
    assert!(fresh_calls > 0);
    assert!(
        reused_calls <= fresh_calls,
        "reused replay asked {reused_calls} symbols, fresh replay {fresh_calls}"
    );
    assert_eq!(reused.subblock_count(), fresh.subblock_count());
    for (&actual, &want) in reused
        .dense_data()
        .unwrap()
        .iter()
        .zip(fresh.dense_data().unwrap())
    {
        assert!((actual - want).abs() <= 1.0e-12 * (1.0 + want.abs()));
    }

    // The plan-cache lease re-attaches the parked destinations; that replay
    // asks no more symbols than the same call with the cache disabled.
    let runtime = tensors[0].runtime().clone();
    let network = injected_network(4, true);
    drop(network.contract(engine, &refs).unwrap());
    symtn::cache::clear();
    provider.reset_symbols();
    let leased = network.contract(engine, &refs).unwrap();
    let leased_calls = provider.symbol_calls.load(Ordering::SeqCst);
    assert_eq!(plan_cache_stats(&runtime).workspace_reuses, 1);
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            enabled: false,
            ..plan_cache_config(&runtime)
        },
    );
    symtn::cache::clear();
    provider.reset_symbols();
    let uncached = network.contract(engine, &refs).unwrap();
    let uncached_calls = provider.symbol_calls.load(Ordering::SeqCst);
    assert!(
        leased_calls <= uncached_calls,
        "leased replay asked {leased_calls} symbols, uncached call {uncached_calls}"
    );
    for (&actual, &want) in leased
        .dense_data()
        .unwrap()
        .iter()
        .zip(uncached.dense_data().unwrap())
    {
        assert!((actual - want).abs() <= 1.0e-12 * (1.0 + want.abs()));
    }
}

#[test]
fn parked_checked_workspace_releases_provider_and_reattaches_only_to_its_runtime() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a checked destination parked by the plan-cache pool keeps no
    // provider, is re-attached to the current call's provider on the next
    // lease, and the parked payload rebinds only to the Runtime it came from.
    use symtn::typed::{__network, Error};
    use symtn_network::plan_cache_stats;

    let runtime = Runtime::builder().build().unwrap();
    let chain = |engine: &mut dyn Engine, provider: &Arc<LieGroupFusionRule>, seed: u64| {
        let leg = GradedSpace::try_new(Arc::clone(provider), [(vec![1], 3)]).unwrap();
        let tensors: Vec<TensorMap<LieGroupFusionRule, f64>> = (0..3)
            .map(|index| {
                TensorMap::rand_with_seed(engine, &runtime, [&leg], [&leg], seed + index).unwrap()
            })
            .collect();
        tensors
    };
    let contract = |engine: &mut dyn Engine, tensors: &[TensorMap<LieGroupFusionRule, f64>]| {
        net(
            &[op(&["a"], &["b"]), op(&["b"], &["c"]), op(&["c"], &["d"])],
            &["a"],
            &["d"],
        )
        .contract(engine, &[&tensors[0], &tensors[1], &tensors[2]])
        .unwrap()
    };

    let first_provider = Arc::new(symtn_core::testing::su_lie_rule(2));
    let first_weak = Arc::downgrade(&first_provider);
    let first = chain(engine, &first_provider, 20);
    drop(contract(engine, &first));
    drop((first, first_provider));
    assert!(first_weak.upgrade().is_none());

    let current_provider = Arc::new(symtn_core::testing::su_lie_rule(2));
    let current = chain(engine, &current_provider, 30);
    let output = contract(engine, &current);
    assert!(std::ptr::eq(output.provider(), current_provider.as_ref()));
    assert_eq!(plan_cache_stats(&runtime).workspace_reuses, 1);
    let expected = current[0]
        .contract(
            engine,
            &current[1],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .contract(
            engine,
            &current[2],
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_same(&output, &expected);

    let other_runtime = Runtime::builder().build().unwrap();
    let parked = __network::detach_runtime(output.clone()).map(|_| ());
    assert!(parked.is_none(), "a shared destination is never parked");
    let authority = current[0].clone();
    let parked = __network::detach_runtime(output).unwrap();
    assert!(matches!(
        parked.can_attach(&other_runtime, &authority),
        Err(Error::RuntimeMismatch)
    ));
    parked.can_attach(&runtime, &authority).unwrap();
    let attached = parked.attach_runtime(&runtime, &authority).unwrap();
    assert!(std::ptr::eq(attached.provider(), current_provider.as_ref()));
    assert_same(&attached, &expected);
}
