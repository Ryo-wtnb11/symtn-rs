use super::*;

mod test_tensor;
pub(crate) use test_tensor::TestTensor;

/// Fixture layout: subblocks packed contiguously in key order. Not a product
/// layout (the only one is the coupled sector matrix); fixtures use it to
/// exercise the arbitrary-strided-view contract of `BlockStructure`.
pub(crate) fn packed_fixture_structure<I, K>(
    rank: usize,
    blocks: I,
) -> Result<BlockStructure, CoreError>
where
    I: IntoIterator<Item = (K, Vec<usize>)>,
    K: Into<BlockKey>,
{
    let mut keys = Vec::new();
    let mut shapes = Vec::new();
    for (key, shape) in blocks {
        keys.push(key.into());
        shapes.push(shape);
    }
    BlockStructure::from_parts(
        SectorStructure::from_keys(rank, keys)?,
        DegeneracyStructure::packed_column_major(rank, shapes)?,
    )
}

/// A direct operand of a contraction entry: its space, stored as itself.
pub(crate) fn direct_side<'a, R, D>(
    space: &'a crate::BoundDynamicFusionMapSpace<R>,
    data: &'a [D],
) -> (
    &'a crate::BoundDynamicFusionMapSpace<R>,
    crate::FusionOperand<'a>,
    &'a [D],
) {
    (space, crate::FusionOperand::direct(space.space()), data)
}

/// The axes of a contraction entry, from a spec without conjugation.
pub(crate) fn entry_axes(
    axes: symtn_operations::TensorContractSpec<'_>,
) -> (&[usize], &[usize], symtn_operations::OutputAxisOrder<'_>) {
    assert!(!axes.lhs_conjugate() && !axes.rhs_conjugate());
    (
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axes.output_permutation(),
    )
}

/// The fusion space of a symmetric fixture.
pub(crate) fn fixture_space<T>(
    tensor: &TestTensor<T>,
) -> Result<DynamicFusionMapSpace, OperationError> {
    Ok(tensor.dynamic_space().clone())
}

/// The eager fusion contraction `dst = alpha * contract(lhs, rhs) + beta *
/// dst` of fixtures through `context`'s dynamic entry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn context_contract_into<BT, BC, R, D>(
    context: &mut TensorContractFusionExecutionContext<D, R::Key, BT, BC, R::Scalar>,
    rule: &R,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, R::Scalar>,
    BC: TensorContractBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    let dst_space = fixture_space(dst)?;
    let lhs_space = fixture_space(lhs)?;
    let rhs_space = fixture_space(rhs)?;
    context.tensorcontract_fusion_dyn_into_raw(
        rule,
        &dst_space,
        dst.data_mut(),
        &lhs_space,
        lhs.data(),
        &rhs_space,
        rhs.data(),
        axes,
        alpha,
        beta,
    )
}

/// Plans the eager fusion contraction of fixtures once, for
/// replay by [`fixture_execute_prepared`] without any cache lookup.
pub(crate) fn fixture_prepare_contract<BT, BC, R, D>(
    context: &mut TensorContractFusionExecutionContext<D, R::Key, BT, BC, R::Scalar>,
    rule: &R,
    dst: &TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    axes: TensorContractSpec<'_>,
) -> Result<StorageContractResolution<R::Scalar>, OperationError>
where
    BT: TreeTransformBackend<D, R::Scalar>,
    BC: TensorContractBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    context.plan_tensorcontract_fusion_dyn_raw(
        rule,
        &fixture_space(dst)?,
        &fixture_space(lhs)?,
        &fixture_space(rhs)?,
        axes,
    )
}

/// Replays a route [`fixture_prepare_contract`] planned:
/// `dst = alpha * contract(lhs, rhs) + beta * dst`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fixture_execute_prepared<BT, BC, RuleKey, C, D>(
    context: &mut TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>,
    resolution: &StorageContractResolution<C>,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    C: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    context.execute_contract_resolution_host(
        resolution,
        (&dst_structure, dst.data_mut()),
        (lhs.structure(), lhs.data()),
        (rhs.structure(), rhs.data()),
        alpha,
        ContractDestinationInit::Axpby(beta),
    )
}

/// [`context_contract_into`] on a fresh default context, for fixtures that
/// need no context reuse.
pub(crate) fn fusion_contract_into<R, D>(
    rule: &R,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    context_contract_into(
        &mut TensorContractFusionExecutionContext::<
            D,
            R::Key,
            DenseTreeTransformOperations,
            DenseTreeTransformOperations,
            R::Scalar,
        >::default(),
        rule,
        dst,
        lhs,
        rhs,
        axes,
        alpha,
        beta,
    )
}

/// The fusion contraction plan of fixture spaces, each checked
/// against `rule` first.
pub(crate) fn fixture_contract_plan<R>(
    rule: &R,
    dst: &DynamicFusionMapSpace,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<FusionContractPlan, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    dst.validate_rule(rule)?;
    lhs.validate_rule(rule)?;
    rhs.validate_rule(rule)?;
    crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw(rule, dst, lhs, rhs, axes)
}

/// Executes a caller-supplied fusion contraction plan step by step (source
/// transforms, core contraction, output transform) on default backends,
/// uncached: the explicit-plan reference for fixtures.
pub(crate) fn plan_contract_into<R, D>(
    rule: &R,
    plan: &FusionContractPlan,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    crate::contract::tensorcontract_fusion_dynamic_plan_into_with(
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TreeTransformWorkspace::default(),
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TensorContractWorkspace::default(),
        rule,
        plan,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

/// Dense (non-fusion) contraction through the compiled structure on the
/// default executor, for fixtures that pin contraction semantics.
pub(crate) fn dense_contract_into<D>(
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    axes: TensorContractSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64>,
{
    let structure = TensorContractStructure::compile_structures(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        axes,
    )?;
    replay_contract_structure_into(
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TensorContractWorkspace::default(),
        &structure,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

/// Replays a compiled dense contraction structure on fixtures
/// through the backend's raw (rank-free) entry.
#[allow(clippy::too_many_arguments)]
pub(crate) fn replay_contract_structure_into<B, D, C>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TensorContractStructure<C>,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TensorContractBackend<D, C>,
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: Copy + num_traits::One,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    backend.tensorcontract_structure_into_raw(
        workspace,
        structure,
        &dst_structure,
        lhs.structure(),
        rhs.structure(),
        dst.data_mut(),
        lhs.data(),
        rhs.data(),
        alpha,
        beta,
    )
}

/// Replays a compiled tree transform on fixtures through the
/// backend's raw (rank-free) entry.
pub(crate) fn replay_into<B, D, C>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, C>,
    D: TreeTransformScalar,
    C: Copy,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    backend.tree_transform_structure_into_raw(
        workspace,
        structure,
        &dst_structure,
        src.structure(),
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

/// Overwrite form of [`replay_into`].
pub(crate) fn replay_overwrite_into<B, D, C>(
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    alpha: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, C>,
    D: TreeTransformScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    backend.tree_transform_structure_overwrite_into_raw(
        workspace,
        structure,
        &dst_structure,
        src.structure(),
        dst.data_mut(),
        src.data(),
        alpha,
        &[],
    )
}

/// The multiplicity-free tree-pair transformer built directly from the
/// public group-plan builder, bypassing the completed-transformer cache: the
/// uncached oracle for fixtures that compare against cached resolution.
pub(crate) fn uncached_tree_pair_structure<R, TDst, TSrc>(
    rule: &R,
    operation: TreeTransformOperation,
    dst: &TestTensor<TDst>,
    src: &TestTensor<TSrc>,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    build_tree_pair_transform_group_plan(rule, operation, src.structure())?
        .compile_shared_structures_with_storage_conjugation(
            std::sync::Arc::clone(dst.structure()),
            std::sync::Arc::clone(src.structure()),
            false,
        )
}

/// The multiplicity-free all-codomain transformer built directly from the
/// public group-plan builder: the uncached all-codomain oracle.
pub(crate) fn uncached_all_codomain_structure<R, TDst, TSrc>(
    rule: &R,
    operation: TreeTransformOperation,
    dst: &TestTensor<TDst>,
    src: &TestTensor<TSrc>,
) -> Result<TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    build_all_codomain_tree_transform_group_plan(rule, operation, src.structure())?
        .compile_shared_structures_with_storage_conjugation(
            std::sync::Arc::clone(dst.structure()),
            std::sync::Arc::clone(src.structure()),
            false,
        )
}

/// The eager multiplicity-free tree transform `dst = beta * dst + alpha *
/// operation(src)` through `context`'s dynamic entry.
pub(crate) fn context_tree_transform_into<B, R, D>(
    context: &mut TreeTransformExecutionContext<D, RuleIdentity, R::Scalar, B>,
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: TreeTransformScalar,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    context.tree_transform_dyn_into(
        rule,
        operation,
        &dst_structure,
        src.structure(),
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

/// Overwrite form of [`context_tree_transform_into`].
pub(crate) fn context_tree_transform_overwrite_into<B, R, D>(
    context: &mut TreeTransformExecutionContext<D, RuleIdentity, R::Scalar, B>,
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    alpha: D,
) -> Result<(), OperationError>
where
    B: TreeTransformBackend<D, R::Scalar>,
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: TreeTransformScalar + RecouplingCoefficientAction<R::Scalar>,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    context.tree_transform_dyn_overwrite_into_ref(
        rule,
        &operation,
        &dst_structure,
        src.structure(),
        dst.data_mut(),
        src.data(),
        alpha,
    )
}

/// [`context_tree_transform_into`] on a fresh default context, for fixtures
/// that need no context reuse.
pub(crate) fn eager_tree_transform_into<R, D>(
    rule: &R,
    operation: TreeTransformOperation,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<R::Scalar>,
{
    context_tree_transform_into(
        &mut TreeTransformExecutionContext::<
            D,
            RuleIdentity,
            R::Scalar,
            DenseTreeTransformOperations,
        >::default(),
        rule,
        operation,
        dst,
        src,
        alpha,
        beta,
    )
}

/// The dynamic fusion trace of fixtures that carry a fusion space.
pub(crate) fn dynamic_trace_spaces<TDst, TSrc>(
    dst: &TestTensor<TDst>,
    src: &TestTensor<TSrc>,
) -> Result<(DynamicFusionMapSpace, DynamicFusionMapSpace), OperationError> {
    Ok((dst.dynamic_space().clone(), src.dynamic_space().clone()))
}

/// The fusion trace `dst = beta * dst + alpha * trace(src)` through the
/// dynamic trace entry.
pub(crate) fn dynamic_tensortrace_fusion_into<R, D>(
    rule: &R,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    axes: TensorTraceAxisSpec<'_>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + RealStructuralCoefficient,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<R::Scalar>
        + strided_kernel::MaybeSendSync,
{
    let (dst_space, src_space) = dynamic_trace_spaces(dst, src)?;
    crate::tensortrace::tensortrace_fusion_dyn_into_raw(
        rule,
        &dst_space,
        dst.data_mut(),
        &src_space,
        src.data(),
        axes,
        alpha,
        beta,
    )
}

/// The compiled fusion trace the dynamic trace entry replays.
pub(crate) fn dynamic_tensortrace_fusion_structure<R, TDst, TSrc>(
    rule: &R,
    dst: &TestTensor<TDst>,
    src: &TestTensor<TSrc>,
    axes: TensorTraceAxisSpec<'_>,
) -> Result<TensorTraceFusionStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + RealStructuralCoefficient,
{
    let (dst_space, src_space) = dynamic_trace_spaces(dst, src)?;
    TensorTraceFusionStructure::compile_fusion_dyn_raw(rule, &dst_space, &src_space, axes)
}

/// Replays a compiled fusion trace on fixtures through the
/// host trace replay.
pub(crate) fn replay_tensortrace_fusion_into<C, D>(
    structure: &TensorTraceFusionStructure<C>,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    C: Copy,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>
        + strided_kernel::MaybeSendSync,
{
    tensortrace_fusion_structure_into_host(
        structure,
        dst.data_mut(),
        src.structure(),
        src.data(),
        alpha,
        beta,
    )
}

use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use std::fmt::Debug;
use std::ops::{Add, Mul};
use symtn_core::{
    multiplicity_free_permute_tree_pair, multiplicity_free_repartition_tree_pair, BlockKey,
    BlockSpec, BlockStructure, BraidingStyleKind, CoreError, DegeneracyStructure,
    FermionParityFusionRule, FusionProductSpace, FusionRule, FusionStyleKind, FusionTreeGroupKey,
    FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey, GenericFArray, GenericFusionSymbols,
    GenericRMatrix, GenericRigidSymbols, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, MultiplicityIndex, Placement, ProductFusionRule, SU2FusionRule,
    SU2Irrep, SectorId, SectorLeg, SectorStructure, SectorVec, U1FusionRule, U1Irrep, Z2FusionRule,
};
use symtn_dense::{DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseWrite, MatrixOp};

/// The single destination of a permutation that is structurally one-to-one.
///
/// Fixtures want the destination tree a permutation lands on, not a proof of
/// which lowering produced it, so they go through the ordinary
/// multiplicity-free entry point and assert the result is a single term.
fn single_permuted_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> (FusionTreePairKey, R::Scalar)
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + std::ops::Add<Output = R::Scalar> + std::ops::Mul<Output = R::Scalar>,
{
    let mut terms = multiplicity_free_permute_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
    )
    .expect("permutation lowering");
    assert_eq!(terms.len(), 1, "fixture permutation is not one-to-one");
    terms.pop().expect("single term")
}

fn fusion_tree_test_key<
    const COD: usize,
    const DOM: usize,
    const COD_DUAL: usize,
    const DOM_DUAL: usize,
>(
    codomain: [usize; COD],
    domain: [usize; DOM],
    coupled: usize,
    codomain_is_dual: [bool; COD_DUAL],
    domain_is_dual: [bool; DOM_DUAL],
) -> FusionTreePairKey {
    FusionTreePairKey::try_pair_from_sector_ids(
        codomain,
        domain,
        coupled,
        codomain_is_dual,
        domain_is_dual,
        [coupled + 100],
        [coupled + 200],
        [coupled + 300],
        [coupled + 400],
    )
    .unwrap()
}

trait TestFusionTreePair {
    fn test_fusion_tree_pair(&self) -> &FusionTreePairKey;
}

impl TestFusionTreePair for BlockKey {
    fn test_fusion_tree_pair(&self) -> &FusionTreePairKey {
        self.as_fusion_tree_pair()
            .expect("test expected a fusion-tree key")
    }
}

impl TestFusionTreePair for FusionTreePairKey {
    fn test_fusion_tree_pair(&self) -> &FusionTreePairKey {
        self
    }
}

fn expect_tree_key(key: &impl TestFusionTreePair) -> FusionTreePairKey {
    key.test_fusion_tree_pair().clone()
}

fn block_offset_for_tree_pair(structure: &BlockStructure, key: &FusionTreePairKey) -> usize {
    structure
        .block(
            structure
                .find_block_index_by_fusion_tree_pair(key)
                .expect("test tree pair must identify a stored block"),
        )
        .unwrap()
        .offset()
}

fn empty_fusion_tree() -> FusionTreeKey {
    empty_fusion_tree_with_coupled(0)
}

fn empty_fusion_tree_with_coupled(coupled: usize) -> FusionTreeKey {
    FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        Vec::<SectorId>::new(),
        SectorId::new(coupled),
        Vec::<bool>::new(),
        Vec::<SectorId>::new(),
        Vec::<MultiplicityIndex>::new(),
    )
    .unwrap()
}

fn all_codomain_fusion_tree_test_key<
    const COD: usize,
    const COD_DUAL: usize,
    const COD_INNER: usize,
    const COD_VERTICES: usize,
>(
    codomain: [usize; COD],
    coupled: usize,
    codomain_is_dual: [bool; COD_DUAL],
    codomain_innerlines: [usize; COD_INNER],
    codomain_vertices: [usize; COD_VERTICES],
) -> FusionTreePairKey {
    all_codomain_fusion_tree_test_key_for_rule(
        &Z2FusionRule,
        codomain,
        coupled,
        codomain_is_dual,
        codomain_innerlines,
        codomain_vertices,
    )
}

fn all_codomain_fusion_tree_test_key_for_rule<
    R,
    const COD: usize,
    const COD_DUAL: usize,
    const COD_INNER: usize,
    const COD_VERTICES: usize,
>(
    rule: &R,
    codomain: [usize; COD],
    coupled: usize,
    codomain_is_dual: [bool; COD_DUAL],
    codomain_innerlines: [usize; COD_INNER],
    codomain_vertices: [usize; COD_VERTICES],
) -> FusionTreePairKey
where
    R: FusionRule,
{
    let tree_pair = FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids_for_rule(
            rule,
            codomain,
            coupled,
            codomain_is_dual,
            codomain_innerlines,
            codomain_vertices,
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            rule,
            Vec::<SectorId>::new(),
            rule.vacuum(),
            Vec::<bool>::new(),
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        )
        .unwrap(),
    );
    tree_pair.validate_for_rule(rule).unwrap();
    tree_pair
}

type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;

fn fz2_u1_su2_tree_pair_fixture() -> (
    FpU1Su2Rule,
    DynamicFusionMapSpace,
    DynamicFusionMapSpace,
    [SectorId; 2],
) {
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let left_sector =
        |parity, charge| left_rule.encode_component_ids(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_component_ids(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };

    let a = sector(odd, 1, 1);
    let b = sector(odd, -1, 1);
    let c0 = sector(even, 0, 0);
    let c1 = sector(even, 0, 2);
    let src_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let dst_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(b, 1)], false),
            SectorLeg::new([(a, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();

    (rule, src_space, dst_space, [c0, c1])
}

fn force_fusion_layout_eviction() {
    // What: exceeds the production fusion-layout entry cap with equal-cost U(1)
    // layouts, forcing any previously inserted layout out of FIFO residency.
    let rule = U1FusionRule;
    for charge in 100_000..108_256 {
        let hom = FusionTreeHomSpace::from_sectors(
            [(U1Irrep::new(charge), 1)],
            [(U1Irrep::new(charge), 1)],
        );
        let _ = hom.fusion_tree_keys(&rule);
    }
}

fn single_transform_coefficient_for_coupled(
    plan: &TreeTransformGroupPlan<f64>,
    coupled: SectorId,
) -> f64 {
    let mut found = None;
    for spec in plan.specs() {
        assert_eq!(spec.src_keys().len(), 1);
        assert_eq!(spec.dst_keys().len(), 1);
        assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
        let dst_coupled = expect_tree_key(&spec.dst_keys()[0]).coupled();
        if dst_coupled == coupled {
            assert!(found.is_none(), "duplicate coefficient for {coupled:?}");
            found = Some(spec.recoupling_coefficients_dst_src()[0]);
        }
    }
    found.unwrap_or_else(|| panic!("missing coefficient for {coupled:?}"))
}

fn expected_single_tree_pair_replay(
    plan: &TreeTransformGroupPlan<f64>,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    initial_dst: &[f64],
    src_data: &[f64],
    alpha: f64,
    beta: f64,
) -> Vec<f64> {
    let mut expected = initial_dst
        .iter()
        .map(|value| beta * value)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        assert_eq!(spec.src_keys().len(), 1);
        assert_eq!(spec.dst_keys().len(), 1);
        assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = block_offset_for_tree_pair(src_structure, src_key);
        let dst_offset = block_offset_for_tree_pair(dst_structure, dst_key);
        expected[dst_offset] +=
            alpha * spec.recoupling_coefficients_dst_src()[0] * src_data[src_offset];
    }
    expected
}

fn column_major_structure_like(structure: &BlockStructure, shape: Vec<usize>) -> BlockStructure {
    let blocks = (0..structure.block_count())
        .map(|index| (structure.block(index).unwrap().key().clone(), shape.clone()));
    packed_fixture_structure(structure.rank(), blocks).unwrap()
}

#[derive(Clone, Copy, Debug)]
struct UniqueZ2Rule;

impl symtn_core::testing::Provider for UniqueZ2Rule {}

impl FusionRule for UniqueZ2Rule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
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

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        vec![SectorId::new((left.id() + right.id()) % 2)].into()
    }
}

impl MultiplicityFreeFusionRule for UniqueZ2Rule {}

impl MultiplicityFreeFusionSymbols for UniqueZ2Rule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for UniqueZ2Rule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
struct UniqueAnyonicRule;

impl symtn_core::testing::Provider for UniqueAnyonicRule {}

impl FusionRule for UniqueAnyonicRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        vec![SectorId::new((left.id() + right.id()) % 2)].into()
    }
}

impl MultiplicityFreeFusionRule for UniqueAnyonicRule {}

impl MultiplicityFreeFusionSymbols for UniqueAnyonicRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        if left.id() == 1 && right.id() == 1 {
            -2.0
        } else {
            1.0
        }
    }
}

impl MultiplicityFreeRigidSymbols for UniqueAnyonicRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
struct AsymmetricAnyonicPointedRule;

impl symtn_core::testing::Provider for AsymmetricAnyonicPointedRule {}

impl FusionRule for AsymmetricAnyonicPointedRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, sector) | (sector, 0) => vec![SectorId::new(sector)].into(),
            (1, 2) | (2, 1) => vec![SectorId::new(3)].into(),
            (3, 1) | (1, 3) => vec![SectorId::new(2)].into(),
            (3, 2) | (2, 3) => vec![SectorId::new(1)].into(),
            _ => vec![SectorId::new((left.id() + right.id()) % 4)].into(),
        }
    }
}

impl MultiplicityFreeFusionRule for AsymmetricAnyonicPointedRule {}

impl MultiplicityFreeFusionSymbols for AsymmetricAnyonicPointedRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        11.0
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        match (left.id(), right.id()) {
            (1, 2) => 5.0,
            (2, 1) => 7.0,
            (3, 2) => 13.0,
            (2, 3) => 17.0,
            (1, 3) => 19.0,
            (3, 1) => 23.0,
            _ => 1.0,
        }
    }
}

impl MultiplicityFreeRigidSymbols for AsymmetricAnyonicPointedRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
struct UniquePlanarRule;

impl symtn_core::testing::Provider for UniquePlanarRule {}

impl FusionRule for UniquePlanarRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::NoBraiding
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        vec![SectorId::new((left.id() + right.id()) % 2)].into()
    }
}

impl MultiplicityFreeFusionRule for UniquePlanarRule {}

impl MultiplicityFreeFusionSymbols for UniquePlanarRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for UniquePlanarRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
struct SimpleSu2Rule;

impl symtn_core::testing::Provider for SimpleSu2Rule {}

impl FusionRule for SimpleSu2Rule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let min = left.id().abs_diff(right.id());
        let max = left.id() + right.id();
        (min..=max).step_by(2).map(SectorId::new).collect()
    }
}

impl MultiplicityFreeFusionRule for SimpleSu2Rule {}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GenericMultiplicityRule;

impl symtn_core::testing::Provider for GenericMultiplicityRule {}

impl FusionRule for GenericMultiplicityRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let channels: Vec<SectorId> = match (left.id(), right.id()) {
            (1, 1) => vec![SectorId::new(0), SectorId::new(1)],
            (0, x) | (x, 0) => vec![SectorId::new(x)],
            _ => Vec::new(),
        };
        channels.into()
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        match (left.id(), right.id(), coupled.id()) {
            (1, 1, 1) => 2,
            _ => usize::from(self.fusion_channels(left, right).contains(&coupled)),
        }
    }
}

impl GenericFusionSymbols for GenericMultiplicityRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let rows = shape.0 * shape.1;
        let columns = shape.2 * shape.3;
        let mut data = vec![0.0; rows * columns];
        for index in 0..rows.min(columns) {
            data[index * columns + index] = 1.0;
        }
        GenericFArray::new(data, shape)
    }

    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        coupled: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let size = if coupled == SectorId::new(1) { 2 } else { 1 };
        let mut data = vec![0.0; size * size];
        for index in 0..size {
            data[index * size + index] = 1.0;
        }
        GenericRMatrix::new(data, size, size)
    }
}

impl GenericRigidSymbols for GenericMultiplicityRule {
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector == SectorId::new(1) {
            (1.0 + 2.0_f64.sqrt()).sqrt()
        } else {
            1.0
        }
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.sqrt_dim_scalar(sector).recip()
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

/// Runs the per-block strided reference driver (`symtn-operations/testing`).
/// The host backend's batched-GEMM path only admits dense float scalars; the
/// integer-dtype replay cases stay on the reference driver.
fn reference_tree_transform<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst: &mut TestTensor<D>,
    src: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let src_structure = std::sync::Arc::clone(src.structure());
    symtn_operations::tree_transform_structure_with_strided_kernel_raw(
        &mut symtn_operations::StridedHostKernelAdapter::default(),
        workspace,
        structure,
        &dst_structure,
        &src_structure,
        dst.data_mut(),
        src.data(),
        alpha,
        beta,
    )
}

fn assert_tree_single_dtype<T>(
    values: Vec<T>,
    fill: T,
    coefficient: T,
    alpha: T,
    beta: T,
    expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let space = vec![2, 2];
    let src = TestTensor::<T>::from_vec(values, space.clone()).unwrap();
    let mut dst = TestTensor::<T>::filled(fill, space).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::single(0, 0, coefficient)],
    )
    .unwrap();
    let mut workspace = TreeTransformWorkspace::default();

    reference_tree_transform(&mut workspace, &structure, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.data(), expected.as_slice());
}

fn assert_tree_multi_dtype<T>(coefficients: Vec<T>, alpha: T, beta: T, fill: T, expected: Vec<T>)
where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let space = vec![4, 2];
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let src = TestTensor::<T>::from_vec_with_structure(
        vec![
            T::one(),
            T::one() + T::one(),
            T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
            T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one() + T::one(),
        ],
        space.clone(),
        src_structure,
    )
    .unwrap();
    let mut dst =
        TestTensor::<T>::from_vec_with_structure(vec![fill; 8], space, dst_structure).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            coefficients,
        )],
    )
    .unwrap();
    let mut workspace = TreeTransformWorkspace::default();

    reference_tree_transform(&mut workspace, &structure, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 8);
    assert_eq!(workspace.destination_len(), 8);
}

fn assert_tree_single_mixed_dtype<D, C>(
    values: Vec<D>,
    fill: D,
    coefficient: C,
    alpha: D,
    beta: D,
    expected: Vec<D>,
) where
    D: TreeTransformScalar + RecouplingCoefficientAction<C> + Clone + Debug,
    C: Copy,
{
    let space = vec![2, 2];
    let src = TestTensor::<D>::from_vec(values, space.clone()).unwrap();
    let mut dst = TestTensor::<D>::filled(fill, space).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::single(0, 0, coefficient)],
    )
    .unwrap();
    let mut workspace = TreeTransformWorkspace::default();

    reference_tree_transform(&mut workspace, &structure, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.data(), expected.as_slice());
}

fn assert_tree_multi_mixed_dtype<D, C>(
    src_values: Vec<D>,
    recoupling_coefficients_dst_src: Vec<C>,
    alpha: D,
    beta: D,
    fill: D,
    expected: Vec<D>,
) where
    D: TreeTransformScalar + RecouplingCoefficientAction<C> + Clone + Debug,
    C: Copy,
{
    let space = vec![4, 2];
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let src =
        TestTensor::<D>::from_vec_with_structure(src_values, space.clone(), src_structure).unwrap();
    let mut dst =
        TestTensor::<D>::from_vec_with_structure(vec![fill; 8], space, dst_structure).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            recoupling_coefficients_dst_src,
        )],
    )
    .unwrap();
    let mut workspace = TreeTransformWorkspace::default();

    reference_tree_transform(&mut workspace, &structure, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 8);
    assert_eq!(workspace.destination_len(), 8);
}

fn assert_tree_multi_tensorkit_orientation_dtype<T>(
    src_values: Vec<T>,
    recoupling_coefficients_dst_src: Vec<T>,
    alpha: T,
    beta: T,
    fill: T,
    expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let src_space = vec![6, 1];
    let dst_space = vec![4, 1];
    let src_structure =
        BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
    let src =
        TestTensor::<T>::from_vec_with_structure(src_values, src_space, src_structure).unwrap();
    let mut dst =
        TestTensor::<T>::from_vec_with_structure(vec![fill; 4], dst_space, dst_structure).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1, 2],
            recoupling_coefficients_dst_src,
        )],
    )
    .unwrap();
    let mut workspace = TreeTransformWorkspace::default();

    reference_tree_transform(&mut workspace, &structure, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

fn assert_tree_multi_tensorkit_orientation_dense_dtype<T>(
    src_values: Vec<T>,
    recoupling_coefficients_dst_src: Vec<T>,
    alpha: T,
    beta: T,
    fill: T,
    expected: Vec<T>,
) where
    T: DenseRecouplingScalar + Clone + Debug,
{
    let src_space = vec![6, 1];
    let dst_space = vec![4, 1];
    let src_structure =
        BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
    let src =
        TestTensor::<T>::from_vec_with_structure(src_values, src_space, src_structure).unwrap();
    let mut dst =
        TestTensor::<T>::from_vec_with_structure(vec![fill; 4], dst_space, dst_structure).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1, 2],
            recoupling_coefficients_dst_src,
        )],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

fn assert_tree_multi_keyed_dtype<T>(
    src_values: Vec<T>,
    recoupling_coefficients_dst_src: Vec<T>,
    expected: Vec<T>,
) where
    T: Copy
        + Clone
        + Debug
        + PartialEq
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + ConjugateValue
        + strided_kernel::MaybeSendSync
        + RecouplingCoefficientAction<T>,
{
    let key10 = BlockKey::opaque([10]);
    let key20 = BlockKey::opaque([20]);
    let key100 = BlockKey::opaque([100]);
    let key200 = BlockKey::opaque([200]);
    let key300 = BlockKey::opaque([300]);
    let src_space = vec![6, 1];
    let dst_space = vec![4, 1];
    let src_structure = packed_fixture_structure(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
    )
    .unwrap();
    let src =
        TestTensor::<T>::from_vec_with_structure(src_values, src_space, src_structure).unwrap();
    let mut dst =
        TestTensor::<T>::from_vec_with_structure(vec![T::zero(); 4], dst_space, dst_structure)
            .unwrap();
    let structure = TreeTransformStructure::compile_keyed_shared_structures(
        std::sync::Arc::clone(dst.structure()),
        std::sync::Arc::clone(src.structure()),
        &[TreeTransformKeyBlockSpec::multi(
            vec![key10, key20],
            vec![key100, key200, key300],
            recoupling_coefficients_dst_src,
        )],
        false,
    )
    .unwrap();
    let mut workspace = TreeTransformWorkspace::default();

    reference_tree_transform(
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        T::one(),
        T::zero(),
    )
    .unwrap();

    assert_eq!(structure.block_count(), 1);
    assert_eq!(dst.data(), expected.as_slice());
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

fn assert_tensorcontract_scalar_dtype<D>(lhs_value: D, rhs_value: D, fill: D, expected: D)
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64> + Clone + Debug,
{
    let lhs_space = vec![1, 1];
    let rhs_space = vec![1, 1];
    let dst_space = vec![1, 1];
    let lhs = TestTensor::<D>::from_vec(vec![lhs_value], lhs_space).unwrap();
    let rhs = TestTensor::<D>::from_vec(vec![rhs_value], rhs_space).unwrap();
    let mut dst = TestTensor::<D>::filled(fill, dst_space).unwrap();

    dense_contract_into(
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        D::one() + D::one(),
        D::one() + D::one() + D::one(),
    )
    .unwrap();

    assert_eq!(dst.data(), &[expected]);
}

fn assert_weighted_tensorcontract_scalar_dtype<D>(lhs_value: D, rhs_value: D, fill: D, expected: D)
where
    D: DenseBlockScalar + RecouplingCoefficientAction<f64> + Clone + Debug,
{
    let lhs_space = vec![1, 1];
    let rhs_space = vec![1, 1];
    let dst_space = vec![1, 1];
    let lhs = TestTensor::<D>::from_vec(vec![lhs_value], lhs_space).unwrap();
    let rhs = TestTensor::<D>::from_vec(vec![rhs_value], rhs_space).unwrap();
    let mut dst = TestTensor::<D>::from_vec(vec![fill], dst_space).unwrap();
    let structure = TensorContractStructure::compile_structures_with_block_specs(
        dst.structure(),
        lhs.structure(),
        rhs.structure(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        &[TensorContractBlockSpec::with_coefficient(0, 0, 0, 0.5)],
    )
    .unwrap();

    replay_contract_structure_into(
        &mut DenseTreeTransformOperations::default_executor(),
        &mut TensorContractWorkspace::default(),
        &structure,
        &mut dst,
        &lhs,
        &rhs,
        D::one() + D::one(),
        D::one() + D::one() + D::one(),
    )
    .unwrap();

    assert_eq!(dst.data(), &[expected]);
}

#[derive(Default)]
struct CountingDenseExecutor {
    dot_general_into_calls: usize,
}

impl DenseExecutor for CountingDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tree transform does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tree transform does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tree transform does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &symtn_dense::DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.dot_general_into_calls += 1;
        assert_eq!(config, &symtn_dense::DenseDotConfig::matmul());

        // This mock pins the TensorKit-style `mul!` boundary only:
        // `buffer_src :: (blocksize, n_src)` times `U^T :: (n_src, n_dst)`
        // into `buffer_dst :: (blocksize, n_dst)`. Numerical GEMM behavior
        // is covered by the DefaultDenseExecutor test.
        let (mut output, lhs, rhs) = match (output, lhs, rhs) {
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                (output, lhs, rhs)
            }
            _ => panic!("counting executor only covers f64 recoupling"),
        };

        assert_eq!(lhs.shape(), &[2, 3]);
        assert_eq!(lhs.strides(), &[1, 2]);
        assert_eq!(lhs.offset(), 0);
        assert_eq!(lhs.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(rhs.shape(), &[3, 2]);
        assert_eq!(rhs.strides(), &[1, 3]);
        assert_eq!(rhs.offset(), 0);
        assert_eq!(rhs.data(), &[10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0]);

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.strides(), &[1, 2]);
        assert_eq!(output.offset(), 0);

        let out_strides = output.strides().to_vec();
        let out_offset = output.offset();
        let out_data = output.data_mut();
        out_data[out_offset] = 5310.0;
        out_data[out_offset + out_strides[0]] = 6420.0;
        out_data[out_offset + out_strides[1]] = 10620.0;
        out_data[out_offset + out_strides[0] + out_strides[1]] = 12840.0;
        Ok(())
    }
}

#[derive(Default)]
struct ContractLayoutCheckingDenseExecutor {
    dot_general_into_calls: usize,
}

impl DenseExecutor for ContractLayoutCheckingDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.dot_general_into_calls += 1;
        assert_eq!(
            config,
            &DenseDotConfig::new(vec![1], vec![1], Vec::new(), Vec::new())
        );
        let (mut output, lhs, rhs) = match (output, lhs, rhs) {
            (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                (output, lhs, rhs)
            }
            _ => panic!("layout-checking executor only covers f64"),
        };

        if lhs.shape() == [2, 3] {
            assert_eq!(lhs.strides(), &[1, 2]);
            assert_eq!(lhs.offset(), 0);
            assert_eq!(rhs.shape(), &[4, 3]);
            assert_eq!(rhs.strides(), &[1, 4]);
            assert_eq!(rhs.offset(), 0);
            assert_eq!(output.shape(), &[2, 4]);
            assert_eq!(output.strides(), &[1, 2]);
            assert_eq!(output.offset(), 0);
            output
                .data_mut()
                .copy_from_slice(&[115.0, 148.0, 124.0, 160.0, 133.0, 172.0, 142.0, 184.0]);
        } else {
            assert_eq!(lhs.shape(), &[4, 3]);
            assert_eq!(lhs.strides(), &[1, 4]);
            assert_eq!(lhs.offset(), 0);
            assert_eq!(rhs.shape(), &[2, 3]);
            assert_eq!(rhs.strides(), &[1, 2]);
            assert_eq!(rhs.offset(), 0);
            assert_eq!(output.shape(), &[4, 2]);
            assert_eq!(output.strides(), &[1, 4]);
            assert_eq!(output.offset(), 0);
            output
                .data_mut()
                .copy_from_slice(&[115.0, 124.0, 133.0, 142.0, 148.0, 160.0, 172.0, 184.0]);
        }
        Ok(())
    }
}

#[derive(Default)]
struct PanicDenseExecutor;

impl DenseExecutor for PanicDenseExecutor {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call svd")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call qr")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<symtn_dense::DenseTensor>, DenseError> {
        unreachable!("tensor contraction does not call eigh")
    }

    fn dot_general_into(
        &mut self,
        _output: DenseWrite<'_>,
        _lhs: DenseRead<'_>,
        _rhs: DenseRead<'_>,
        _config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        panic!("replay structure mismatch must be rejected before dense execution")
    }
}

mod adjoint_memo;

mod adjoint_view;

mod alternating_bicharacter;

mod contract_dense;

mod contract_fusion;

mod subset_cross_operations;

mod tensortrace;

mod tensorkit_scaling;

mod tree_transform_exec;

mod tree_transform_plan;
