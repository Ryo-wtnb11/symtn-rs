//! The Host compile entry of the storage (device) contraction route, and —
//! with the `cuda` feature, on a real device — its executor replaying the
//! Host `DynamicTree` artifact (#1345).
//!
//! The device tests force every orientation and axis-order candidate through
//! the test-only plan builder, which the typed API cannot reach, and compare
//! the device replay of each artifact with the Host replay of the same
//! artifact. Independent oracles for the typed lowering live in
//! `symtn/tests/typed_contract_host_oracle.rs` and `typed_cuda_contract.rs`.

use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    ProductFusionRule, SU2FusionRule, SU2Irrep, SectorId, SectorLeg, U1FusionRule, U1Irrep,
};
use symtn_operations::{OutputAxisOrder, TensorContractSpec};

use crate::contract::dynamic::{
    compile_dynamic_tree_execution_artifact, execute_dynamic_tree_execution_artifact,
    DynamicTreeExecutionArtifact,
};
use crate::contract::fusion::FusionContractOrientation;
use crate::contract::scratch::DynamicFusionScratchWorkspace;
use crate::tree_context::TreeTransformExecutionContext;
use crate::{
    BoundDynamicFusionMapSpace, DenseRecouplingScalar, DenseTreeTransformOperations, FusionOperand,
    RecouplingCoefficientAction, RuleIdentity,
};

type Context<D> = crate::TensorContractFusionExecutionContext<D, RuleIdentity>;

fn space<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    codomain: Vec<SectorLeg>,
    domain: Vec<SectorLeg>,
) -> BoundDynamicFusionMapSpace<R> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ),
    )
    .unwrap()
}

fn u1_leg(dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 2),
        ],
        dual,
    )
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
            (SU2Irrep::from_twice_spin(2).sector_id(), 1),
        ],
        false,
    )
}

/// A rank-5 × rank-4 U(1) contraction on a codomain leg against a domain leg
/// and a domain leg against a codomain leg, contracted out of order, with a
/// permuted output: source transforms on both sides and an output transform.
struct Case<R> {
    lhs: BoundDynamicFusionMapSpace<R>,
    rhs: BoundDynamicFusionMapSpace<R>,
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

impl<R: MultiplicityFreeRigidSymbols<Scalar = f64>> Case<R> {
    fn dst(&self) -> BoundDynamicFusionMapSpace<R> {
        BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &self.lhs,
            &self.rhs,
            &self.lhs_axes,
            &self.rhs_axes,
            OutputAxisOrder::from_axes(&self.output_axes),
        )
        .unwrap()
    }

    fn axes(&self) -> TensorContractSpec<'_> {
        TensorContractSpec::new(
            &self.lhs_axes,
            &self.rhs_axes,
            OutputAxisOrder::from_axes(&self.output_axes),
        )
    }
}

fn u1_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v(), v()], vec![v(), v()]),
        rhs: space(&provider, vec![v(), v()], vec![v(), v()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    }
}

fn su2_case() -> Case<SU2FusionRule> {
    let provider = Arc::new(SU2FusionRule);
    let s = su2_leg;
    Case {
        lhs: space(&provider, vec![s(), s()], vec![s(), s()]),
        rhs: space(&provider, vec![s(), s()], vec![s()]),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![0, 1],
        output_axes: vec![1, 0, 2],
    }
}

#[derive(Default)]
struct CountingDense {
    inner: symtn_dense::DefaultDenseExecutor,
    submissions: Vec<usize>,
}

impl symtn_dense::DenseExecutor for CountingDense {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: symtn_dense::DenseRead<'_>,
    ) -> Result<Vec<symtn_dense::DenseTensor>, symtn_dense::DenseError> {
        symtn_dense::DenseExecutor::svd(&mut self.inner, engine, input)
    }
    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: symtn_dense::DenseRead<'_>,
    ) -> Result<Vec<symtn_dense::DenseTensor>, symtn_dense::DenseError> {
        symtn_dense::DenseExecutor::qr(&mut self.inner, engine, input)
    }
    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: symtn_dense::DenseRead<'_>,
    ) -> Result<Vec<symtn_dense::DenseTensor>, symtn_dense::DenseError> {
        symtn_dense::DenseExecutor::eigh(&mut self.inner, engine, input)
    }
    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: symtn_dense::DenseWrite<'_>,
        lhs: symtn_dense::DenseRead<'_>,
        rhs: symtn_dense::DenseRead<'_>,
        config: &symtn_dense::DenseDotConfig,
    ) -> Result<(), symtn_dense::DenseError> {
        symtn_dense::DenseExecutor::dot_general_into(
            &mut self.inner,
            engine,
            output,
            lhs,
            rhs,
            config,
        )
    }
    fn matmul_batch_axpby_into(
        &mut self,
        engine: &mut dyn Engine,
        output: symtn_dense::DenseWrite<'_>,
        lhs: symtn_dense::DenseRead<'_>,
        rhs: symtn_dense::DenseRead<'_>,
        jobs: &[symtn_dense::DenseGemmBatchJob],
        runs: &[usize],
        alpha: symtn_dense::DenseScalar,
        beta: symtn_dense::DenseScalar,
    ) -> Result<(), symtn_dense::DenseError> {
        self.submissions.push(jobs.len());
        symtn_dense::DenseExecutor::matmul_batch_axpby_into(
            &mut self.inner,
            engine,
            output,
            lhs,
            rhs,
            jobs,
            runs,
            alpha,
            beta,
        )
    }
}

fn physical_oracle<R>(case: &Case<R>, lhs: &[f64], rhs: &[f64]) -> (Vec<usize>, Vec<f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + symtn_core::PhysicalFusionBasis<Scalar = f64>,
{
    let (a_shape, a) =
        crate::expand_physical_host(crate::BoundDynamicTensorRef::try_new(&case.lhs, lhs).unwrap())
            .unwrap();
    let (b_shape, b) =
        crate::expand_physical_host(crate::BoundDynamicTensorRef::try_new(&case.rhs, rhs).unwrap())
            .unwrap();
    let strides = |shape: &[usize]| {
        shape
            .iter()
            .scan(1usize, |stride, &dim| {
                let current = *stride;
                *stride *= dim;
                Some(current)
            })
            .collect::<Vec<_>>()
    };
    let a_stride = strides(&a_shape);
    let b_stride = strides(&b_shape);
    let a_open = (0..a_shape.len())
        .filter(|axis| !case.lhs_axes.contains(axis))
        .collect::<Vec<_>>();
    let b_open = (0..b_shape.len())
        .filter(|axis| !case.rhs_axes.contains(axis))
        .collect::<Vec<_>>();
    let joint_shape = a_open
        .iter()
        .map(|&axis| a_shape[axis])
        .chain(b_open.iter().map(|&axis| b_shape[axis]))
        .collect::<Vec<_>>();
    let output_shape = case
        .output_axes
        .iter()
        .map(|&axis| joint_shape[axis])
        .collect::<Vec<_>>();
    let output_strides = strides(&output_shape);
    let contracted_shape = case
        .lhs_axes
        .iter()
        .map(|&axis| a_shape[axis])
        .collect::<Vec<_>>();
    let contracted_strides = strides(&contracted_shape);
    let mut joint = vec![0; joint_shape.len()];
    let mut result = vec![0.0; output_shape.iter().product()];
    for (linear, value) in result.iter_mut().enumerate() {
        for (position, &axis) in case.output_axes.iter().enumerate() {
            joint[axis] = (linear / output_strides[position]) % output_shape[position];
        }
        let a_base = a_open
            .iter()
            .enumerate()
            .map(|(position, &axis)| joint[position] * a_stride[axis])
            .sum::<usize>();
        let b_base = b_open
            .iter()
            .enumerate()
            .map(|(position, &axis)| joint[a_open.len() + position] * b_stride[axis])
            .sum::<usize>();
        for contracted in 0..contracted_shape.iter().product::<usize>() {
            let mut ai = a_base;
            let mut bi = b_base;
            for (position, (&a_axis, &b_axis)) in
                case.lhs_axes.iter().zip(&case.rhs_axes).enumerate()
            {
                let coordinate =
                    (contracted / contracted_strides[position]) % contracted_shape[position];
                ai += coordinate * a_stride[a_axis];
                bi += coordinate * b_stride[b_axis];
            }
            *value += a[ai] * b[bi];
        }
    }
    (output_shape, result)
}

fn su2_rank5_case() -> Case<SU2FusionRule> {
    let provider = Arc::new(SU2FusionRule);
    let s = su2_leg;
    Case {
        lhs: space(&provider, vec![s(), s(), s()], vec![s(), s()]),
        rhs: space(&provider, vec![s(), s()], vec![s(), s()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    }
}

#[test]
fn dynamic_artifact_replays_host_members() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let case = su2_rank5_case();
    let destination = case.dst();
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            engine,
            &destination,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    let super::resolution::ContractRoute::DynamicTree(artifact) = resolution.route else {
        panic!("fixture must select DynamicTree");
    };
    assert!(artifact
        .test_lhs_transform()
        .0
        .has_pack_gemm_scatter_blocks());
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let member_len = destination.space().required_len().unwrap();
    let mut workspace = super::HostContractMembersWorkspace::default();
    let mut independent = super::HostContractMembersWorkspace::default();
    let mut first_bytes = 0;
    let mut b2_submissions = 0;
    for (pass, members) in [1, 2, 2, 17, 1].into_iter().enumerate() {
        let lhs = (0..members)
            .flat_map(|i| host_data(case.lhs.space(), 3 + 2 * i))
            .collect::<Vec<_>>();
        let rhs = (0..members)
            .flat_map(|i| host_data(case.rhs.space(), 7 + 4 * i))
            .collect::<Vec<_>>();
        let mut actual = vec![f64::NAN; members * member_len];
        let current = if pass == 4 {
            &mut independent
        } else {
            &mut workspace
        };
        let mut dense = DenseTreeTransformOperations::new(CountingDense::default());
        let mut backend = DenseTreeTransformOperations::new(CountingDense::default());
        let mut backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
        if members == 2 {
            let original = actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert!(
                super::route_host::execute_dynamic_tree_execution_artifact_members_host(
                    engine,
                    &mut dense,
                    &mut backend,
                    &mut backend_workspace,
                    &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
                    &artifact,
                    destination.space().structure(),
                    current,
                    &mut actual,
                    &lhs,
                    &rhs[..rhs.len() - 1],
                    members,
                )
                .is_err()
            );
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                original
            );
            assert!(dense.dense().submissions.is_empty());
            assert!(backend.dense_mut().submissions.is_empty());
            let short_len = actual.len() - 1;
            assert!(
                super::route_host::execute_dynamic_tree_execution_artifact_members_host(
                    engine,
                    &mut dense,
                    &mut backend,
                    &mut backend_workspace,
                    &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
                    &artifact,
                    destination.space().structure(),
                    current,
                    &mut actual[..short_len],
                    &lhs,
                    &rhs,
                    members,
                )
                .is_err()
            );
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                original
            );
            assert!(dense.dense().submissions.is_empty());
            assert!(backend.dense_mut().submissions.is_empty());
        }
        let builds_before = current.core_replay_builds();
        let engine =
            &mut symtn_operations::host_pool::test_engine(if members == 1 { 2 } else { 1 });
        super::route_host::execute_dynamic_tree_execution_artifact_members_host(
            engine,
            &mut dense,
            &mut backend,
            &mut backend_workspace,
            &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
            &artifact,
            destination.space().structure(),
            current,
            &mut actual,
            &lhs,
            &rhs,
            members,
        )
        .unwrap();
        assert_eq!(
            current.core_replay_builds(),
            builds_before + usize::from(pass != 2)
        );
        let submissions = dense.dense().submissions.len() + backend.dense_mut().submissions.len();
        eprintln!("#1647 SU(2) B={members}: Host dense submissions={submissions}, transform jobs={}, core jobs={}", dense.dense().submissions.iter().sum::<usize>(), backend.dense_mut().submissions.iter().sum::<usize>());
        if members == 2 {
            b2_submissions = submissions;
        }
        if members == 17 {
            assert_eq!(submissions, b2_submissions);
        }
        if members > 1 {
            assert!(dense
                .dense()
                .submissions
                .iter()
                .any(|&jobs| jobs >= members));
            assert!(backend
                .dense_mut()
                .submissions
                .iter()
                .any(|&jobs| jobs >= members));
        }
        assert!(actual.iter().all(|x| x.is_finite()));
        if pass == 0 {
            first_bytes = current.retained_bytes();
        }
        if pass == 3 {
            assert!(current.retained_bytes() < 100 * first_bytes.max(1));
        }
        for member in 0..members {
            let left = &lhs[member * lhs_len..(member + 1) * lhs_len];
            let right = &rhs[member * rhs_len..(member + 1) * rhs_len];
            let observed = &actual[member * member_len..(member + 1) * member_len];
            let expected = eager_host(&case, left, right);
            for (&found, &want) in observed.iter().zip(&expected) {
                assert!((found - want).abs() < 1e-9 * (1.0 + want.abs()));
            }
            // Why the dense oracle at B = 1 too: eager and B = 1 member
            // replays are one executor, so eager alone proves nothing there.
            if members <= 2 || member == 0 || member + 1 == members {
                let (oracle_shape, oracle) = physical_oracle(&case, left, right);
                let (found_shape, found) = crate::expand_physical_host(
                    crate::BoundDynamicTensorRef::try_new(&destination, observed).unwrap(),
                )
                .unwrap();
                assert_eq!(found_shape, oracle_shape);
                for (&value, &want) in found.iter().zip(&oracle) {
                    assert!((value - want).abs() < 1e-8 * (1.0 + want.abs()));
                }
            }
        }
    }
}

#[test]
fn dynamic_artifact_complex_members_match_eager() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    use num_complex::Complex64;
    let case = su2_rank5_case();
    let destination = case.dst();
    let resolution = Context::<Complex64>::default()
        .compile_storage_contract_resolution(
            engine,
            &destination,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    let super::resolution::ContractRoute::DynamicTree(artifact) = resolution.route else {
        panic!("fixture must select DynamicTree");
    };
    let mut workspace = super::HostContractMembersWorkspace::default();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let dst_len = destination.space().required_len().unwrap();
    for members in [1, 2, 17] {
        let values = |space, base| {
            (0..members)
                .flat_map(|member| {
                    host_data(space, base + member * 3)
                        .into_iter()
                        .enumerate()
                        .map(|(i, x)| Complex64::new(x, (i as f64 * 0.13 + x).cos()))
                })
                .collect::<Vec<_>>()
        };
        let lhs = values(case.lhs.space(), 5);
        let rhs = values(case.rhs.space(), 9);
        let mut actual = vec![Complex64::new(f64::NAN, f64::NAN); members * dst_len];
        super::route_host::execute_dynamic_tree_execution_artifact_members_host(
            engine,
            &mut DenseTreeTransformOperations::default(),
            &mut DenseTreeTransformOperations::default(),
            &mut crate::contract::backend::TensorContractWorkspace::default(),
            &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
            &artifact,
            destination.space().structure(),
            &mut workspace,
            &mut actual,
            &lhs,
            &rhs,
            members,
        )
        .unwrap();
        for member in 0..members {
            let expected = eager_host(
                &case,
                &lhs[member * lhs_len..(member + 1) * lhs_len],
                &rhs[member * rhs_len..(member + 1) * rhs_len],
            );
            for (&found, &want) in actual[member * dst_len..(member + 1) * dst_len]
                .iter()
                .zip(&expected)
            {
                assert!((found - want).norm() < 1e-9 * (1.0 + want.norm()));
            }
        }
    }
}

#[test]
fn dynamic_artifact_overwrites_inactive_u1_members() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (_, case, _) = overwrite_cases().pop().unwrap();
    let destination = case.dst();
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            engine,
            &destination,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    let super::resolution::ContractRoute::DynamicTree(artifact) = resolution.route else {
        panic!("fixture must select DynamicTree");
    };
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let dst_len = destination.space().required_len().unwrap();
    let lhs = [
        host_data(case.lhs.space(), 2),
        host_data(case.lhs.space(), 5),
    ]
    .concat();
    let rhs = [
        host_data(case.rhs.space(), 7),
        host_data(case.rhs.space(), 11),
    ]
    .concat();
    let mut actual = vec![f64::NAN; 2 * dst_len];
    super::route_host::execute_dynamic_tree_execution_artifact_members_host(
        engine,
        &mut DenseTreeTransformOperations::default(),
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &artifact,
        destination.space().structure(),
        &mut super::HostContractMembersWorkspace::default(),
        &mut actual,
        &lhs,
        &rhs,
        2,
    )
    .unwrap();
    for member in 0..2 {
        let expected = eager_host(
            &case,
            &lhs[member * lhs_len..(member + 1) * lhs_len],
            &rhs[member * rhs_len..(member + 1) * rhs_len],
        );
        for (&found, &want) in actual[member * dst_len..(member + 1) * dst_len]
            .iter()
            .zip(&expected)
        {
            assert!(found.is_finite());
            assert!((found - want).abs() < 1e-9 * (1.0 + want.abs()));
        }
    }
}

/// Replays `members` copies of `case` through `artifact` into a NaN-poisoned
/// caller destination, checks every value against the eager replay, and
/// returns how many inactive-core fills the replay performed.
fn replay_members_checked(
    case: &Case<U1FusionRule>,
    artifact: &DynamicTreeExecutionArtifact<f64>,
    workspace: &mut super::HostContractMembersWorkspace<f64>,
    members: usize,
    salt: usize,
) -> usize {
    try_replay_members_checked(case, artifact, workspace, members, salt).unwrap()
}

fn try_replay_members_checked(
    case: &Case<U1FusionRule>,
    artifact: &DynamicTreeExecutionArtifact<f64>,
    workspace: &mut super::HostContractMembersWorkspace<f64>,
    members: usize,
    salt: usize,
) -> Result<usize, symtn_operations::OperationError> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let destination = case.dst();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let dst_len = destination.space().required_len().unwrap();
    let lhs = (0..members)
        .flat_map(|i| host_data(case.lhs.space(), salt + 2 * i))
        .collect::<Vec<_>>();
    let rhs = (0..members)
        .flat_map(|i| host_data(case.rhs.space(), salt + 5 + 3 * i))
        .collect::<Vec<_>>();
    let mut actual = vec![f64::NAN; members * dst_len];
    let before = workspace.core_inactive_fills();
    super::route_host::execute_dynamic_tree_execution_artifact_members_host(
        engine,
        &mut DenseTreeTransformOperations::default(),
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        artifact,
        destination.space().structure(),
        workspace,
        &mut actual,
        &lhs,
        &rhs,
        members,
    )?;
    for member in 0..members {
        let expected = eager_host(
            case,
            &lhs[member * lhs_len..(member + 1) * lhs_len],
            &rhs[member * rhs_len..(member + 1) * rhs_len],
        );
        for (&found, &want) in actual[member * dst_len..(member + 1) * dst_len]
            .iter()
            .zip(&expected)
        {
            assert!(found.is_finite());
            assert!((found - want).abs() < 1e-9 * (1.0 + want.abs()));
        }
    }
    Ok(workspace.core_inactive_fills() - before)
}

fn members_artifact(case: &Case<U1FusionRule>) -> Arc<DynamicTreeExecutionArtifact<f64>> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            engine,
            &case.dst(),
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    let super::resolution::ContractRoute::DynamicTree(artifact) = resolution.route else {
        panic!("fixture must select DynamicTree");
    };
    artifact
}

#[test]
fn a_retained_workspace_core_destination_is_not_rezeroed_on_warm_replay() {
    let mut cases = overwrite_cases();
    let (_, workspace_dst, _) = cases.pop().unwrap();
    let (_, caller_dst, _) = cases.remove(1);
    let workspace_artifact = members_artifact(&workspace_dst);
    let caller_artifact = members_artifact(&caller_dst);
    assert!(workspace_artifact.test_has_core_dst());
    assert!(!caller_artifact.test_has_core_dst());

    // Workspace-owned core destination: only a new replay (cold, B change,
    // plan change) fills the inactive blocks; an unchanged replay does not.
    let mut ws = super::HostContractMembersWorkspace::default();
    assert_eq!(
        replay_members_checked(&workspace_dst, &workspace_artifact, &mut ws, 2, 3),
        1
    );
    assert_eq!(
        replay_members_checked(&workspace_dst, &workspace_artifact, &mut ws, 2, 9),
        0
    );
    assert_eq!(
        replay_members_checked(&workspace_dst, &workspace_artifact, &mut ws, 2, 14),
        0
    );
    assert_eq!(
        replay_members_checked(&workspace_dst, &workspace_artifact, &mut ws, 3, 4),
        1
    );
    assert_eq!(
        replay_members_checked(&workspace_dst, &workspace_artifact, &mut ws, 3, 8),
        0
    );
    // Plan change: a caller-destination artifact replaces the core, then the
    // original returns and must refill because its replay is new again.
    assert_eq!(
        replay_members_checked(&caller_dst, &caller_artifact, &mut ws, 3, 6),
        1
    );
    assert_eq!(
        replay_members_checked(&workspace_dst, &workspace_artifact, &mut ws, 3, 2),
        1
    );
    assert_eq!(
        replay_members_checked(&workspace_dst, &workspace_artifact, &mut ws, 3, 5),
        0
    );

    // Caller-owned destination (identity output): every call fills, even
    // for an unchanged replay into freshly poisoned memory.
    let mut ws = super::HostContractMembersWorkspace::default();
    for salt in [3, 9, 14] {
        assert_eq!(
            replay_members_checked(&caller_dst, &caller_artifact, &mut ws, 2, salt),
            1
        );
    }
}

#[test]
fn a_failed_replay_into_a_dirtied_workspace_destination_rezeroes_on_retry() {
    // Why: a new plan installs its replay, then a later step fails; the
    // workspace destination may hold the previous plan's active results in
    // this plan's inactive blocks, so the retry must fill them again.
    let (_, case, _) = overwrite_cases().pop().unwrap();
    let artifact = members_artifact(&case);
    let mut ws = super::HostContractMembersWorkspace::default();
    assert_eq!(replay_members_checked(&case, &artifact, &mut ws, 2, 3), 1);
    assert_eq!(replay_members_checked(&case, &artifact, &mut ws, 2, 9), 0);
    let fresh = members_artifact(&case);
    ws.poison_dst_then_fail();
    assert!(try_replay_members_checked(&case, &fresh, &mut ws, 2, 5).is_err());
    assert_eq!(replay_members_checked(&case, &fresh, &mut ws, 2, 5), 1);
    assert_eq!(replay_members_checked(&case, &fresh, &mut ws, 2, 7), 0);
}

#[test]
fn dynamic_artifact_rejects_malformed_twisted_members_before_writes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (_, case) = fermionic_cases().into_iter().next().unwrap();
    let lhs_one = host_data(case.lhs.space(), 3);
    let rhs_one = host_data(case.rhs.space(), 7);
    let artifact =
        crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
            .into_iter()
            .flat_map(|candidate| {
                orientations().map(move |orientation| (candidate.clone(), orientation))
            })
            .map(|(candidate, orientation)| {
                forced_artifact(&case, &candidate, orientation, &lhs_one, &rhs_one).0
            })
            .find(|artifact| artifact.requires_source_twist())
            .expect("fermionic fixture needs a twisted artifact");
    let destination = case.dst();
    let mut dst = vec![f64::NAN; 2 * destination.space().required_len().unwrap()];
    let lhs = [lhs_one.as_slice(), lhs_one.as_slice()].concat();
    let rhs = rhs_one;
    let before = dst.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut dense = DenseTreeTransformOperations::new(CountingDense::default());
    let mut backend = DenseTreeTransformOperations::new(CountingDense::default());
    let error = super::route_host::execute_dynamic_tree_execution_artifact_members_host(
        engine,
        &mut dense,
        &mut backend,
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &artifact,
        destination.space().structure(),
        &mut super::HostContractMembersWorkspace::default(),
        &mut dst,
        &lhs,
        &rhs,
        2,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        crate::OperationError::ElementCountMismatch { .. }
    ));
    assert_eq!(dst.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), before);
    assert!(dense.dense().submissions.is_empty());
    assert!(backend.dense_mut().submissions.is_empty());
}

#[test]
fn rank_five_su2_member_transform_matches_physical_permutation_and_ordinary_replay() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let s = su2_leg;
    let case = Case {
        lhs: space(&provider, vec![s(), s(), s()], vec![s(), s()]),
        rhs: space(&provider, vec![s(), s()], vec![s(), s()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    };
    let destination = case.dst();
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            engine,
            &destination,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    let super::resolution::ContractRoute::DynamicTree(artifact) = resolution.route else {
        panic!("noncomposition SU(2) fixture must select DynamicTree");
    };
    let (transform, dst_structure, src_structure, transformed_space) =
        artifact.test_lhs_transform();
    assert!(transform.has_pack_gemm_scatter_blocks());
    let plan = super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw_canonical(
        destination.provider(),
        destination.space(),
        case.lhs.space(),
        case.rhs.space(),
        case.axes(),
    )
    .unwrap();
    let order: Vec<_> = plan
        .lhs_transform()
        .codomain_permutation()
        .iter()
        .chain(plan.lhs_transform().domain_permutation())
        .copied()
        .collect();
    assert_eq!(order, [0, 2, 4, 1, 3]);
    let transformed_bound =
        BoundDynamicFusionMapSpace::from_derived_like(&case.lhs, transformed_space.clone())
            .unwrap();
    let source_len = src_structure.required_len().unwrap();
    let destination_len = dst_structure.required_len().unwrap();
    let source = [
        host_data(case.lhs.space(), 11),
        host_data(case.lhs.space(), 19),
    ]
    .concat();
    assert_eq!(source.len(), 2 * source_len);
    let mut batched = vec![f64::NAN; 2 * destination_len];
    symtn_operations::tree_transform_members_overwrite_raw(
        engine,
        &mut symtn_operations::StridedHostKernelAdapter::default(),
        &mut symtn_dense::DefaultDenseExecutor::new(),
        &mut symtn_operations::TreeTransformWorkspace::default(),
        transform,
        dst_structure,
        src_structure,
        &mut batched,
        &source,
        2,
        1,
        &[],
    )
    .unwrap();
    for member in 0..2 {
        let (source_shape, source_physical) = crate::expand_physical_host(
            crate::BoundDynamicTensorRef::try_new(
                &case.lhs,
                &source[member * source_len..(member + 1) * source_len],
            )
            .unwrap(),
        )
        .unwrap();
        let (target_shape, target_physical) = crate::expand_physical_host(
            crate::BoundDynamicTensorRef::try_new(
                &transformed_bound,
                &batched[member * destination_len..(member + 1) * destination_len],
            )
            .unwrap(),
        )
        .unwrap();
        let strides = |shape: &[usize]| {
            shape
                .iter()
                .scan(1usize, |stride, &dim| {
                    let current = *stride;
                    *stride *= dim;
                    Some(current)
                })
                .collect::<Vec<_>>()
        };
        let source_strides = strides(&source_shape);
        let target_strides = strides(&target_shape);
        assert_eq!(
            target_shape,
            order
                .iter()
                .map(|&axis| source_shape[axis])
                .collect::<Vec<_>>()
        );
        let scale = source_physical
            .iter()
            .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
        let tolerance = 32.0 * (source_len as f64).sqrt() * f64::EPSILON * scale;
        for (linear, &actual) in target_physical.iter().enumerate() {
            let source_linear = order
                .iter()
                .enumerate()
                .map(|(position, &axis)| {
                    ((linear / target_strides[position]) % target_shape[position])
                        * source_strides[axis]
                })
                .sum::<usize>();
            assert!(
                (actual - source_physical[source_linear]).abs() <= tolerance,
                "physical member {member}, element {linear}: {actual} != {}",
                source_physical[source_linear]
            );
        }
        let mut expected = vec![f64::NAN; destination_len];
        symtn_operations::tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut symtn_operations::StridedHostKernelAdapter::default(),
            &mut symtn_dense::DefaultDenseExecutor::new(),
            &mut symtn_operations::TreeTransformWorkspace::default(),
            transform,
            dst_structure,
            src_structure,
            &mut expected,
            &source[member * source_len..(member + 1) * source_len],
            1.0,
            &[],
            1,
        )
        .unwrap();
        // One linear source transform cannot sum more terms into an entry
        // than its source payload length (docs/testing_numerics.md).
        let scale = expected
            .iter()
            .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
        let tolerance = 32.0 * (source_len as f64).sqrt() * f64::EPSILON * scale;
        for (index, (&actual, &reference)) in batched
            [member * destination_len..(member + 1) * destination_len]
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert!(
                (actual - reference).abs() <= tolerance,
                "member {member} element {index}: {actual} != {reference}"
            );
        }
    }
}

/// `lhs` already has its contracted leg as its whole domain: under LhsRhs
/// its source transform is the identity and it is read in place; the output
/// is permuted, so an output transform remains.
fn lhs_identity_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v()], vec![v()]),
        rhs: space(&provider, vec![v(), v()], vec![v()]),
        lhs_axes: vec![2],
        rhs_axes: vec![1],
        output_axes: vec![3, 1, 0, 2],
    }
}

/// `rhs` already has its contracted leg as its whole codomain.
fn rhs_identity_case() -> Case<U1FusionRule> {
    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    Case {
        lhs: space(&provider, vec![v(), v()], vec![v()]),
        rhs: space(&provider, vec![u1_leg(true)], vec![v(), v()]),
        lhs_axes: vec![0],
        rhs_axes: vec![0],
        output_axes: vec![1, 0, 2, 3],
    }
}

/// The artifact of one forced axis-order candidate and orientation — the
/// test-only plan builder reaches both, the typed API only the scorer's
/// choice — together with its Host replay over `lhs`/`rhs` and the core-right
/// contracted axes of its plan.
fn forced_artifact<R, D>(
    case: &Case<R>,
    candidate: &crate::contract::fusion::ContractAxisOrderCandidate,
    orientation: FusionContractOrientation,
    lhs: &[D],
    rhs: &[D],
) -> (DynamicTreeExecutionArtifact<f64>, Vec<D>, Vec<usize>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = case.lhs.provider();
    let dst = case.dst();
    let plan =
        crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
            rule,
            dst.space(),
            case.lhs.space(),
            case.rhs.space(),
            case.axes(),
            candidate,
            orientation,
        )
        .unwrap();
    let mut tree_context: TreeTransformExecutionContext<D, RuleIdentity, f64, _> =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let artifact =
        compile_dynamic_tree_execution_artifact::<symtn_core::MultiplicityFreeAdmissionMode, _>(
            engine,
            tree_context.planning(),
            rule,
            crate::contract::encoded_layout_primer::<R>,
            &plan,
            dst.space(),
            case.lhs.space(),
            case.lhs.space().structure(),
            case.rhs.space(),
            case.rhs.space().structure(),
        )
        .unwrap();
    let mut host = vec![D::zero(); dst.space().required_len().unwrap()];
    execute_dynamic_tree_execution_artifact(
        engine,
        &mut tree_context,
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &mut DynamicFusionScratchWorkspace::default(),
        &artifact,
        dst.space().structure(),
        &mut host,
        lhs,
        rhs,
        D::one(),
        D::zero(),
    )
    .unwrap();
    let core_right_contracting = plan.core_axes().as_spec().rhs_contracting_axes().to_vec();
    (artifact, host, core_right_contracting)
}

fn orientations() -> [FusionContractOrientation; 2] {
    [
        FusionContractOrientation::LhsRhs,
        FusionContractOrientation::RhsLhs,
    ]
}

type FermionU1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
type FermionSu2 = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

const EVEN: SectorId = SectorId::new(0);
const ODD: SectorId = SectorId::new(1);

/// fZ2 x U(1), degeneracy > 1, odd sectors of both charge signs. A dual leg
/// stores the conjugate charges (the typed `GradedSpace::try_dual`).
fn fermion_u1_leg(rule: &FermionU1, dual: bool) -> SectorLeg {
    let charge = |q: i32| U1Irrep::new(if dual { -q } else { q }).sector_id();
    SectorLeg::new(
        [
            (rule.encode_component_ids(EVEN, charge(0)), 2),
            (rule.encode_component_ids(ODD, charge(1)), 2),
            (rule.encode_component_ids(ODD, charge(-1)), 1),
            (rule.encode_component_ids(EVEN, charge(1)), 1),
        ],
        dual,
    )
}

/// fZ2 (x) SU(2): the fermionic rule whose transforms recouple (Multi blocks).
fn fermion_su2_leg(rule: &FermionSu2, dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (
                rule.encode_component_ids(EVEN, SU2Irrep::from_twice_spin(0).sector_id()),
                2,
            ),
            (
                rule.encode_component_ids(ODD, SU2Irrep::from_twice_spin(1).sector_id()),
                2,
            ),
            (
                rule.encode_component_ids(EVEN, SU2Irrep::from_twice_spin(2).sector_id()),
                1,
            ),
        ],
        dual,
    )
}

/// The geometry of `symtn/tests/contract_cases::fermionic_general`: rank 4
/// against rank 4 over mixed sides, B's codomain leg `v*` and B's domain leg
/// `u` contracted (dual on B's codomain side, and dual after bending from
/// its domain side unless `u` is itself dual — then θ is mixed within one
/// coupled sector).
fn fermionic_general_case<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    leg: impl Fn(bool) -> SectorLeg,
    u_dual: bool,
) -> Case<R> {
    Case {
        lhs: space(
            provider,
            vec![leg(false), leg(u_dual)],
            vec![leg(false), leg(false)],
        ),
        rhs: space(
            provider,
            vec![leg(false), leg(true)],
            vec![leg(false), leg(u_dual)],
        ),
        lhs_axes: vec![1, 0],
        rhs_axes: vec![3, 1],
        output_axes: vec![2, 0, 3, 1],
    }
}

/// The canonical `mul!` form with B's codomain `(v, v*)`: θ varies within one
/// coupled-sector matrix (`contract_cases::fermionic_canonical_nonuniform`).
fn fermionic_canonical_case<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    leg: impl Fn(bool) -> SectorLeg,
) -> Case<R> {
    Case {
        lhs: space(
            provider,
            vec![leg(false), leg(false)],
            vec![leg(false), leg(true)],
        ),
        rhs: space(provider, vec![leg(false), leg(true)], vec![leg(false)]),
        lhs_axes: vec![2, 3],
        rhs_axes: vec![0, 1],
        output_axes: vec![0, 1, 2],
    }
}

/// Which operands the `LhsRhs` layout of a [`fermionic_twist_role_cases`]
/// fixture already copies, in `(A, B)` order.
#[derive(Clone, Copy, Debug)]
enum RoleCase {
    ACopied,
    Canonical,
    BCopied,
    BothCopied,
}

/// The geometries of `symtn/tests/contract_cases::fermionic_twist_roles`:
/// one per operand-copy case of TensorKit's twist choice, each with a dual
/// leg among B's contracted legs, and A smaller than B where size decides.
fn fermionic_twist_role_cases<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    leg: impl Fn(bool) -> SectorLeg,
) -> [(RoleCase, Case<R>); 4] {
    let v = || leg(false);
    let v_dual = || leg(true);
    [
        (
            RoleCase::ACopied,
            Case {
                lhs: space(provider, vec![v(), v()], vec![v()]),
                rhs: space(provider, vec![v_dual()], vec![v(), v()]),
                lhs_axes: vec![1],
                rhs_axes: vec![0],
                output_axes: vec![0, 1, 2, 3],
            },
        ),
        (
            RoleCase::Canonical,
            Case {
                lhs: space(provider, vec![v()], vec![v(), v_dual()]),
                rhs: space(provider, vec![v(), v_dual()], vec![v(), v()]),
                lhs_axes: vec![1, 2],
                rhs_axes: vec![0, 1],
                output_axes: vec![0, 1, 2],
            },
        ),
        (
            RoleCase::BCopied,
            Case {
                lhs: space(provider, vec![v(), v()], vec![v_dual()]),
                rhs: space(provider, vec![v()], vec![v(), v()]),
                lhs_axes: vec![2],
                rhs_axes: vec![1],
                output_axes: vec![0, 1, 2, 3],
            },
        ),
        (
            RoleCase::BothCopied,
            Case {
                lhs: space(provider, vec![v(), v()], vec![v()]),
                rhs: space(provider, vec![v(), v_dual()], vec![v(), v()]),
                lhs_axes: vec![1, 0],
                rhs_axes: vec![3, 1],
                output_axes: vec![0, 1, 2],
            },
        ),
    ]
}

fn fermionic_cases() -> Vec<(&'static str, Case<FermionU1>)> {
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let leg = |dual| fermion_u1_leg(&provider, dual);
    let mut cases = vec![
        ("fZ2xU1 both", fermionic_general_case(&provider, leg, false)),
        ("fZ2xU1 mixed", fermionic_general_case(&provider, leg, true)),
        ("fZ2xU1 canonical", fermionic_canonical_case(&provider, leg)),
    ];
    let names = [
        "fZ2xU1 A copied",
        "fZ2xU1 canonical small A",
        "fZ2xU1 B copied",
        "fZ2xU1 both copied small A",
    ];
    cases.extend(
        names
            .into_iter()
            .zip(fermionic_twist_role_cases(&provider, leg))
            .map(|(name, (_, case))| (name, case)),
    );
    cases
}

fn fermionic_su2_cases() -> Vec<(&'static str, Case<FermionSu2>)> {
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let leg = |dual| fermion_su2_leg(&provider, dual);
    let mut cases = vec![
        (
            "fZ2xSU2 both",
            fermionic_general_case(&provider, leg, false),
        ),
        (
            "fZ2xSU2 mixed",
            fermionic_general_case(&provider, leg, true),
        ),
        (
            "fZ2xSU2 canonical",
            fermionic_canonical_case(&provider, leg),
        ),
    ];
    let names = [
        "fZ2xSU2 A copied",
        "fZ2xSU2 canonical small A",
        "fZ2xSU2 B copied",
        "fZ2xSU2 both copied small A",
    ];
    cases.extend(
        names
            .into_iter()
            .zip(fermionic_twist_role_cases(&provider, leg))
            .map(|(name, (_, case))| (name, case)),
    );
    cases
}

fn host_data(space: &crate::DynamicFusionMapSpace, salt: usize) -> Vec<f64> {
    (0..space.required_len().unwrap())
        .map(|index| ((index * 7 + salt) as f64 * 0.37 + 0.1).sin())
        .collect()
}

/// The per-block twist of one physical operand's transformed source,
/// recomputed from that space alone (`lhs` names the side) by the twist
/// compiler, as the sorted `(offset, θ)` of its non-empty twisted blocks.
fn physical_twist<R>(
    case: &Case<R>,
    artifact: &DynamicTreeExecutionArtifact<f64>,
    orientation: FusionContractOrientation,
    contracting: &[usize],
) -> Vec<(usize, f64)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let core_right_is_lhs = orientation == FusionContractOrientation::RhsLhs;
    let lhs = artifact.twists_lhs();
    super::dynamic::compile_contract_twist(
        case.lhs.provider(),
        artifact.physical_core_space(lhs),
        artifact.physical_core_space(core_right_is_lhs).homspace(),
        lhs != core_right_is_lhs,
        contracting,
    )
    .unwrap()
}

/// The eager Host contraction of `case` over the `lhs`/`rhs` payloads: the
/// production answer every forced artifact must reproduce.
fn eager_host<R, D>(case: &Case<R>, lhs: &[D], rhs: &[D]) -> Vec<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let dst = case.dst();
    let mut out = vec![D::zero(); dst.space().required_len().unwrap()];
    Context::<D>::default()
        .tensorcontract_prelowered_into(
            engine,
            &dst,
            &mut out,
            FusionOperand::direct(case.lhs.space()),
            lhs,
            FusionOperand::direct(case.rhs.space()),
            rhs,
            case.axes(),
            D::one(),
            crate::ContractDestinationInit::Axpby(D::zero()),
        )
        .unwrap();
    out
}

/// Every forced candidate and orientation of one fermionic case:
///
/// - the destination-scale list is the per-block twist of the *physical*
///   operand the artifact twists — core-right or, when only core-left is
///   already copied (or is the smaller), core-left — recomputed from that
///   space and the core-right dual flags alone;
/// - the compile-time uniform-per-Multi assertion passed;
/// - the Host replay of the forced artifact equals the eager Host
///   `contract`, so every forced orientation is tied to the production
///   answer, not only to a device replay of the same artifact.
///
/// Returns, per orientation (`[LhsRhs, RhsLhs]`), whether any artifact
/// twisted, and whether any twisted core-right transform had a Multi block.
fn check_destination_scales<R>(case: &Case<R>, what: &str) -> ([bool; 2], bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let lhs = host_data(case.lhs.space(), 1);
    let rhs = host_data(case.rhs.space(), 2);
    let eager = eager_host(case, &lhs, &rhs);
    let scale = eager
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    assert!(scale > 1e-3, "{what}: vacuous all-zero eager result");
    let (mut twisted, mut multi) = ([false; 2], false);
    for candidate in
        crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
    {
        for (slot, orientation) in orientations().into_iter().enumerate() {
            let where_ = format!("{what} {candidate:?} {orientation:?}");
            let (artifact, host, contracting) =
                forced_artifact(case, &candidate, orientation, &lhs, &rhs);
            let scales = artifact.source_twist_destination_scales();
            assert_eq!(
                scales,
                physical_twist(case, &artifact, orientation, &contracting).as_slice(),
                "{where_}: not the twisted physical operand's twist"
            );
            assert!(scales.windows(2).all(|pair| pair[0].0 < pair[1].0));
            assert!(scales.iter().all(|&(_, theta)| theta == -1.0));
            assert_eq!(host.len(), eager.len(), "{where_}");
            for (index, (&forced, &production)) in host.iter().zip(&eager).enumerate() {
                assert!(
                    (forced - production).abs() <= 1e-12 * (1.0 + scale),
                    "{where_}: element {index} is {forced}, eager Host {production}"
                );
            }
            if !scales.is_empty() {
                twisted[slot] = true;
                multi |= artifact
                    .twisted_transform_structure()
                    .blocks()
                    .iter()
                    .any(|block| {
                        matches!(block, symtn_operations::TreeTransformBlock::Multi { .. })
                    });
            }
        }
    }
    (twisted, multi)
}

#[test]
fn the_destination_scale_list_is_the_hosts_per_block_twist_in_both_orientations() {
    let mut report = Vec::new();
    let mut any_multi = false;
    let mut record = |what: &str, (twisted, multi): ([bool; 2], bool)| {
        report.push(format!(
            "{what}: LhsRhs {} RhsLhs {}",
            twisted[0], twisted[1]
        ));
        any_multi |= multi;
        twisted
    };
    let mut u1 = Vec::new();
    for (what, case) in fermionic_cases() {
        u1.push(record(what, check_destination_scales(&case, what)));
    }
    let mut su2 = Vec::new();
    for (what, case) in fermionic_su2_cases() {
        su2.push(record(what, check_destination_scales(&case, what)));
    }
    // What: every fixture twists under LhsRhs, and for each provider some
    // fixture also twists under RhsLhs, so neither orientation is covered
    // only vacuously. (With `u` not dual the lhs's contracted legs are all
    // non-dual: under RhsLhs that fixture carries no twist, and its forced
    // replays still equal the eager Host result above.)
    for twisted in [&u1, &su2] {
        assert!(twisted.iter().all(|t| t[0]), "{report:?}");
        assert!(twisted.iter().any(|t| t[0] && t[1]), "{report:?}");
    }
    // What: the uniform-per-Multi assertion was really exercised.
    assert!(any_multi, "no twisted transform recoupled");
}

#[test]
fn a_scale_list_that_varies_within_one_multi_block_is_an_internal_error() {
    let lhs_rhs = |case: &Case<FermionSu2>| {
        (
            host_data(case.lhs.space(), 1),
            host_data(case.rhs.space(), 2),
        )
    };
    let mut checked = false;
    for (what, case) in fermionic_su2_cases() {
        let (lhs, rhs) = lhs_rhs(&case);
        for candidate in
            crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
        {
            for orientation in orientations() {
                let (artifact, _, _) = forced_artifact(&case, &candidate, orientation, &lhs, &rhs);
                let structure = artifact.twisted_transform_structure();
                let scales = artifact.source_twist_destination_scales();
                assert!(
                    super::dynamic::validate_uniform_multi_scales(structure, scales).is_ok(),
                    "{what}"
                );
                let layouts = structure.layouts();
                let Some(first) = structure.blocks().iter().find_map(|block| match *block {
                    symtn_operations::TreeTransformBlock::Multi {
                        dst_layout_start,
                        dst_count,
                        ..
                    } => {
                        let live: Vec<usize> = (dst_layout_start..dst_layout_start + dst_count)
                            .filter(|&index| layouts.entry(index).element_count != 0)
                            .map(|index| usize::try_from(layouts.entry(index).offset).unwrap())
                            .collect();
                        (live.len() >= 2).then(|| live[0])
                    }
                    symtn_operations::TreeTransformBlock::Single { .. } => None,
                }) else {
                    continue;
                };
                // Flip θ of one destination layout of that Multi block only.
                let mut tampered = scales.to_vec();
                match tampered.binary_search_by_key(&first, |&(offset, _)| offset) {
                    Ok(index) => {
                        tampered.remove(index);
                    }
                    Err(index) => tampered.insert(index, (first, -1.0)),
                }
                let error = super::dynamic::validate_uniform_multi_scales(structure, &tampered)
                    .unwrap_err();
                assert!(
                    matches!(error, crate::OperationError::InvalidArgument { message } if message.contains("not uniform")),
                    "{error:?}"
                );
                checked = true;
            }
        }
    }
    assert!(
        checked,
        "no fixture had a Multi block with two live destinations"
    );
}

/// TensorKit's `blas_contract!` choice of the twisted operand
/// (tensoroperations.jl:398-409 @cfaa073), in the `LhsRhs` orientation:
/// the planner's cost, the compiled artifact's borrowing and the Host
/// replay's source transforms all agree that the twist rides on the operand
/// already copied (or on the smaller one) and adds no materialization.
fn check_twist_role<R>(role: RoleCase, case: &Case<R>, what: &str) -> (usize, usize)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = case.lhs.provider();
    let dst = case.dst();
    let e_a = case.lhs.space().required_len().unwrap();
    let e_b = case.rhs.space().required_len().unwrap();
    if matches!(role, RoleCase::Canonical | RoleCase::BothCopied) {
        assert!(e_a < e_b, "{what}: A must be the smaller operand");
    }
    let facts = crate::contract::fusion::prepare_tensorcontract_fusion_candidate_facts_dyn_raw(
        rule,
        dst.space(),
        case.lhs.space(),
        case.rhs.space(),
        case.axes(),
    )
    .unwrap();
    // What: this candidate is also the one the production scorer selects.
    let cheapest = facts.iter().map(|f| f.total_materialized_elements()).min();
    let facts = &facts[0];
    assert_eq!(
        cheapest,
        Some(facts.total_materialized_elements()),
        "{what}"
    );
    assert_eq!(
        facts.orientation(),
        FusionContractOrientation::LhsRhs,
        "{what}"
    );
    // (A copied by layout, B copied by layout, A twisted, total when the twist
    // always materializes B).
    let (a_copied, b_copied, twist_a, before) = match role {
        RoleCase::ACopied => (true, false, true, e_a + e_b),
        RoleCase::Canonical => (false, false, true, e_b),
        RoleCase::BCopied => (false, true, false, e_b),
        RoleCase::BothCopied => (true, true, true, e_a + e_b),
    };
    assert_eq!(facts.lhs_exact_identity_borrowable(), !a_copied, "{what}");
    assert_eq!(facts.rhs_exact_identity_borrowable(), !b_copied, "{what}");
    assert!(facts.lhs_requires_twist() == twist_a, "{what}");
    assert!(facts.rhs_requires_twist() != twist_a, "{what}");
    let lhs_len = if a_copied || twist_a { e_a } else { 0 };
    let rhs_len = if b_copied || !twist_a { e_b } else { 0 };
    assert_eq!(facts.lhs_materialized_elements(), lhs_len, "{what}");
    assert_eq!(facts.rhs_materialized_elements(), rhs_len, "{what}");

    let lhs = host_data(case.lhs.space(), 1);
    let rhs = host_data(case.rhs.space(), 2);
    let candidate =
        crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes).remove(0);
    let (artifact, host, _) = forced_artifact(
        case,
        &candidate,
        FusionContractOrientation::LhsRhs,
        &lhs,
        &rhs,
    );
    assert!(artifact.requires_source_twist(), "{what}");
    assert_eq!(artifact.twists_lhs(), twist_a, "{what}");
    assert_eq!(
        artifact.borrowed_sources(),
        (lhs_len == 0, rhs_len == 0),
        "{what}"
    );

    let mut tree_context = TreeTransformExecutionContext::<f64, RuleIdentity>::new(
        DenseTreeTransformOperations::default_executor(),
    );
    let mut replayed = vec![0.0; dst.space().required_len().unwrap()];
    super::dynamic::execute_dynamic_tree_execution_artifact(
        engine,
        &mut tree_context,
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &mut DynamicFusionScratchWorkspace::default(),
        &artifact,
        dst.space().structure(),
        &mut replayed,
        &lhs,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();

    let eager = eager_host(case, &lhs, &rhs);
    let scale = eager
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()));
    for (index, ((&forced, &replayed), &production)) in
        host.iter().zip(&replayed).zip(&eager).enumerate()
    {
        assert!(
            (forced - production).abs() <= 1e-12 * (1.0 + scale)
                && (replayed - production).abs() <= 1e-12 * (1.0 + scale),
            "{what}: element {index} is {forced}/{replayed}, eager Host {production}"
        );
    }
    let mut first_workspace = super::HostContractMembersWorkspace::default();
    let mut first_dense = DenseTreeTransformOperations::new(CountingDense::default());
    let mut first_backend = DenseTreeTransformOperations::new(CountingDense::default());
    let mut first_backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
    let mut first_result = vec![f64::NAN; eager.len()];
    super::route_host::execute_dynamic_tree_execution_artifact_members_host(
        engine,
        &mut first_dense,
        &mut first_backend,
        &mut first_backend_workspace,
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &artifact,
        dst.space().structure(),
        &mut first_workspace,
        &mut first_result,
        &lhs,
        &rhs,
        1,
    )
    .unwrap();
    assert_eq!(first_backend.dense_mut().submissions.len(), 1);
    let jobs_per_member = first_backend.dense_mut().submissions[0];
    let mut member_workspace = super::HostContractMembersWorkspace::default();
    let mut dense = DenseTreeTransformOperations::new(CountingDense::default());
    let mut backend = DenseTreeTransformOperations::new(CountingDense::default());
    let mut backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
    let lhs_members = [lhs.as_slice(), lhs.as_slice()].concat();
    let rhs_members = [rhs.as_slice(), rhs.as_slice()].concat();
    let mut member_result = vec![f64::NAN; 2 * eager.len()];
    super::route_host::execute_dynamic_tree_execution_artifact_members_host(
        engine,
        &mut dense,
        &mut backend,
        &mut backend_workspace,
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        &artifact,
        dst.space().structure(),
        &mut member_workspace,
        &mut member_result,
        &lhs_members,
        &rhs_members,
        2,
    )
    .unwrap();
    assert!(artifact.requires_source_twist());
    assert_eq!(backend.dense_mut().submissions, [2 * jobs_per_member]);
    for (actual, expected) in member_result.iter().zip(eager.iter().cycle()) {
        assert!((actual - expected).abs() <= 1e-12 * (1.0 + scale), "{what}");
    }
    (lhs_len + rhs_len, before)
}

#[test]
fn the_fermionic_twist_rides_on_the_operand_already_copied_or_the_smaller() {
    let fu1 = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let fsu2 = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let mut totals = Vec::new();
    for (role, case) in fermionic_twist_role_cases(&fu1, |dual| fermion_u1_leg(&fu1, dual)) {
        totals.push((
            role,
            check_twist_role(role, &case, &format!("fZ2xU1 {role:?}")),
        ));
    }
    for (role, case) in fermionic_twist_role_cases(&fsu2, |dual| fermion_su2_leg(&fsu2, dual)) {
        totals.push((
            role,
            check_twist_role(role, &case, &format!("fZ2xSU2 {role:?}")),
        ));
    }
    // What: twisting the operand already copied (#1351) removes the E_B-sized
    // materialization exactly where only A is copied (A copied: E_A + E_B -> E_A; canonical with A smaller:
    // E_B -> E_A); B copied and both copied are unchanged.
    for (role, (after, before)) in totals {
        match role {
            RoleCase::ACopied | RoleCase::Canonical => assert!(after < before, "{role:?}"),
            RoleCase::BCopied | RoleCase::BothCopied => assert_eq!(after, before, "{role:?}"),
        }
    }
}

/// A storage GEMM the Host storage-direct entry must reject before reaching.
struct RejectingGemm;

impl symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>
    for RejectingGemm
{
    fn matmul_range_into(
        &mut self,
        _dst: &mut Vec<f64>,
        _dst_offset: usize,
        _lhs: &Vec<f64>,
        _lhs_offset: usize,
        _rhs: &Vec<f64>,
        _rhs_offset: usize,
        _rows: usize,
        _contracted: usize,
        _cols: usize,
    ) -> Result<(), crate::OperationError> {
        panic!("the storage-direct entry must reject a nonuniform twist before any GEMM")
    }
}

/// Why the device lifts the canonical non-uniform-θ case to DynamicTree: no
/// per-job GEMM alpha expresses a twist that varies within one coupled-sector
/// matrix. The core route declines it for every executor, so the
/// storage-direct entry, which has no other route, reports it unsupported
/// before any GEMM; the compiled artifact carries the twist.
#[test]
fn a_canonical_nonuniform_twist_leaves_the_device_core_route_for_dynamic_tree() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (_, case) = fermionic_cases().remove(2);
    let dst = case.dst();
    let (lhs, rhs) = (
        FusionOperand::direct(case.lhs.space()),
        FusionOperand::direct(case.rhs.space()),
    );
    assert!(
        crate::try_compile_storage_contract_core_route::<crate::DirectCoreExecutor, _>(
            &dst,
            lhs,
            rhs,
            case.axes()
        )
        .unwrap()
        .hit()
        .is_none()
    );
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(engine, &dst, lhs, rhs, case.axes())
        .unwrap();
    assert!(resolution.is_dynamic_tree());
    assert!(resolution.requires_source_twist());

    let lhs_data = host_data(case.lhs.space(), 1);
    let rhs_data = host_data(case.rhs.space(), 2);
    let mut out = vec![0.0; dst.space().required_len().unwrap()];
    let error = crate::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
        &mut RejectingGemm,
        &dst,
        &mut out,
        lhs,
        &lhs_data,
        rhs,
        &rhs_data,
        case.axes(),
    )
    .unwrap_err();
    assert!(
        matches!(
            error,
            crate::OperationError::UnsupportedTensorContractScope { message }
                if message.contains("canonical fully-direct")
        ),
        "{error:?}"
    );
}

#[test]
fn identity_source_fixtures_compile_a_dynamic_tree() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for case in [lhs_identity_case(), rhs_identity_case()] {
        let resolution = Context::<f64>::default()
            .compile_storage_contract_resolution(
                engine,
                &case.dst(),
                FusionOperand::direct(case.lhs.space()),
                FusionOperand::direct(case.rhs.space()),
                case.axes(),
            )
            .unwrap();
        assert!(resolution.is_dynamic_tree());
    }
}

#[test]
fn canonical_axes_keep_the_direct_core_route_and_others_compile_one_dynamic_tree() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let case = u1_case();
    let dst = case.dst();
    let mut context = Context::<f64>::default();
    let general = context
        .compile_storage_contract_resolution(
            engine,
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    // What: arbitrary axes take the Host DynamicTree artifact, bosonic so no
    // core-right twist.
    assert!(general.is_dynamic_tree());
    assert!(!general.requires_source_twist());
    let su2 = su2_case();
    let su2_resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            engine,
            &su2.dst(),
            FusionOperand::direct(su2.lhs.space()),
            FusionOperand::direct(su2.rhs.space()),
            su2.axes(),
        )
        .unwrap();
    assert!(su2_resolution.is_dynamic_tree());

    let provider = Arc::new(U1FusionRule);
    let v = || u1_leg(false);
    let lhs = space(&provider, vec![v(), v()], vec![v()]);
    let rhs = space(&provider, vec![v()], vec![v(), v()]);
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &[2],
        &[0],
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let canonical = context
        .compile_storage_contract_resolution(
            engine,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&[2], &[0], OutputAxisOrder::identity()),
        )
        .unwrap();
    // What: the TensorKit `mul!` form stays on the fully-direct core route.
    assert!(!canonical.is_dynamic_tree());

    // What: a lazy adjoint with arbitrary axes compiles the prelowered
    // artifact, never the Host's dense `Structure` route.
    let adjoint_bound = lhs.adjoint_view().unwrap();
    let rhs3 = space(&provider, vec![v(), v()], vec![v()]);
    let output = [3, 0, 1, 2];
    let lazy_dst = crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &adjoint_bound,
        &rhs3,
        &[1],
        &[0],
        OutputAxisOrder::from_axes(&output),
    )
    .unwrap();
    let lazy = context
        .compile_storage_contract_resolution(
            engine,
            &lazy_dst,
            FusionOperand::adjoint(lhs.space()),
            FusionOperand::direct(rhs3.space()),
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                OutputAxisOrder::from_axes(&output),
                true,
                false,
            ),
        )
        .unwrap();
    assert!(lazy.is_dynamic_tree());
    assert!(!lazy.requires_source_twist());
}

/// The geometry of `symtn/tests/contract_cases::su2_core_form_lazy_cases`:
/// the Host eager entry takes TensorKit's `copyC` (the planner); the
/// copyC-free storage ladder compiles the prelowered `DynamicTree` artifact.
#[test]
fn a_self_dual_core_form_lazy_contraction_is_host_copy_c_and_storage_dynamic_tree() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let s = su2_leg;
    // Lazy lhs: `X^H` against `rhs` over `X^H`'s whole domain, output bent.
    let x = space(&provider, vec![s(), s()], vec![s(), s()]);
    let rhs = space(&provider, vec![s(), s()], vec![s()]);
    // Lazy rhs: `lhs` against `Y^H` over `Y^H`'s whole codomain, output swapped.
    let lhs = space(&provider, vec![s()], vec![s(), s()]);
    let y = space(&provider, vec![s()], vec![s(), s()]);
    let cases = [
        (
            x.adjoint_view().unwrap(),
            FusionOperand::adjoint(x.space()),
            rhs.clone(),
            FusionOperand::direct(rhs.space()),
            vec![2, 3],
            vec![0, 1],
            vec![2, 0, 1],
            (true, false),
        ),
        (
            lhs.clone(),
            FusionOperand::direct(lhs.space()),
            y.adjoint_view().unwrap(),
            FusionOperand::adjoint(y.space()),
            vec![1, 2],
            vec![0, 1],
            vec![1, 0],
            (false, true),
        ),
    ];
    for (lhs_logical, lhs_operand, rhs_logical, rhs_operand, lhs_axes, rhs_axes, output, conj) in
        cases
    {
        let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &lhs_logical,
            &rhs_logical,
            &lhs_axes,
            &rhs_axes,
            OutputAxisOrder::from_axes(&output),
        )
        .unwrap();
        let axes = TensorContractSpec::new_with_conjugation(
            &lhs_axes,
            &rhs_axes,
            OutputAxisOrder::from_axes(&output),
            conj.0,
            conj.1,
        );
        let lhs_data = vec![0.5; lhs_operand.storage_space().required_len().unwrap()];
        let rhs_data = vec![0.25; rhs_operand.storage_space().required_len().unwrap()];
        let mut out = vec![0.0; dst.space().required_len().unwrap()];
        let mut context = Context::<f64>::default();
        context
            .tensorcontract_prelowered_into(
                engine,
                &dst,
                &mut out,
                lhs_operand,
                &lhs_data,
                rhs_operand,
                &rhs_data,
                axes,
                1.0,
                crate::ContractDestinationInit::Axpby(0.0),
            )
            .unwrap();
        // What: the Host took `copyC` (no core, no transformed source).
        assert!(
            !context.last_resolution_is_core() && context.last_resolution_orientation().is_none(),
            "{output:?}"
        );
        assert!(out.iter().any(|&value| value != 0.0));
        let storage = context
            .compile_storage_contract_resolution(engine, &dst, lhs_operand, rhs_operand, axes)
            .unwrap();
        assert!(storage.is_dynamic_tree(), "{output:?}");
    }
}

#[test]
fn a_fermionic_dual_contracted_leg_on_a_transformed_operand_reports_the_twist() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(FermionParityFusionRule);
    let odd_even = |dual| SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], dual);
    // `rhs` codomain leg 1 is dual: contracting it is a supertrace twist on
    // the core-right operand.
    let lhs = space(
        &provider,
        vec![odd_even(false), odd_even(false)],
        vec![odd_even(false)],
    );
    let rhs = space(
        &provider,
        vec![odd_even(false), odd_even(true)],
        vec![odd_even(false)],
    );
    let lhs_axes = [2, 0];
    let rhs_axes = [0, 1];
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &lhs_axes,
        &rhs_axes,
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let mut context = Context::<f64>::default();
    let resolution = context
        .compile_storage_contract_resolution(
            engine,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&lhs_axes, &rhs_axes, OutputAxisOrder::identity()),
        )
        .unwrap();
    // What: the twist travels in the artifact, which the device replays
    // with destination scales.
    assert!(resolution.is_dynamic_tree());
    assert!(resolution.requires_source_twist());
}

/// Why the device executor needs no zero fill for a core job with
/// `contracted == 0`: no admitted space has one. A zero degeneracy is dropped
/// from the leg, so every operand block has a non-empty contracted extent and
/// a coupled sector with nothing to contract has no job at all — its
/// destination block is then an inactive block, zeroed by the core zeroing
/// rule instead.
#[test]
fn a_zero_degeneracy_sector_never_yields_a_zero_extent_core_job() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let leg = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 1),
                (U1Irrep::new(1).sector_id(), 1),
            ],
            false,
        )
    };
    let empty = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 0),
                (U1Irrep::new(1).sector_id(), 2),
            ],
            false,
        )
    };
    let case = Case {
        lhs: space(&provider, vec![leg(), leg()], vec![empty()]),
        rhs: space(&provider, vec![empty()], vec![leg()]),
        lhs_axes: vec![2],
        rhs_axes: vec![0],
        output_axes: vec![1, 0, 2],
    };
    let no_zero_extent = |space: &crate::DynamicFusionMapSpace| {
        let structure = space.structure();
        (0..structure.block_count())
            .all(|index| !structure.block(index).unwrap().shape().contains(&0))
    };
    assert!(no_zero_extent(case.lhs.space()));
    assert!(no_zero_extent(case.rhs.space()));
    let dst = case.dst();
    let resolution = Context::<f64>::default()
        .compile_storage_contract_resolution(
            engine,
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap();
    assert!(resolution.is_dynamic_tree());
}

/// `wide` carries charge 2 that `narrow` lacks, so a destination block
/// coupled at charge 2 has no GEMM job.
fn wide_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 2),
            (U1Irrep::new(2).sector_id(), 1),
        ],
        false,
    )
}

fn narrow_leg(dual: bool) -> SectorLeg {
    let sign = if dual { -1 } else { 1 };
    SectorLeg::new(
        [
            (U1Irrep::new(0).sector_id(), 2),
            (U1Irrep::new(sign).sector_id(), 1),
        ],
        dual,
    )
}

/// One geometry per way the device completes a destination it did not
/// allocate (`contract_into`), with the core plan's inactive
/// destination-block count when the core GEMMs write that destination
/// directly (`None`: an output transform writes it).
fn overwrite_cases() -> Vec<(&'static str, Case<U1FusionRule>, Option<usize>)> {
    let provider = Arc::new(U1FusionRule);
    let (v, w) = (wide_leg, || narrow_leg(false));
    let case = |lhs, rhs, lhs_axes: &[usize], rhs_axes: &[usize], output_axes: &[usize]| Case {
        lhs,
        rhs,
        lhs_axes: lhs_axes.to_vec(),
        rhs_axes: rhs_axes.to_vec(),
        output_axes: output_axes.to_vec(),
    };
    vec![
        (
            "core route, inactive block",
            case(
                space(&provider, vec![v(), v()], vec![w()]),
                space(&provider, vec![w()], vec![v()]),
                &[2],
                &[0],
                &[0, 1, 2],
            ),
            Some(1),
        ),
        (
            "transformed lhs, identity output, inactive block",
            case(
                space(&provider, vec![v()], vec![w(), v()]),
                space(&provider, vec![w()], vec![v()]),
                &[1],
                &[0],
                &[0, 1, 2],
            ),
            Some(1),
        ),
        (
            "transformed rhs, identity output, fully covered",
            case(
                space(&provider, vec![v(), v()], vec![w()]),
                space(&provider, vec![v()], vec![narrow_leg(true)]),
                &[2],
                &[1],
                &[0, 1, 2],
            ),
            Some(0),
        ),
        (
            "output transform over an inactive core block",
            case(
                space(&provider, vec![v(), v()], vec![w()]),
                space(&provider, vec![w()], vec![v()]),
                &[2],
                &[0],
                &[1, 0, 2],
            ),
            None,
        ),
    ]
}

#[test]
fn each_way_a_device_overwrite_completes_its_destination_has_a_fixture() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // Why: the device gate replays these into a NaN-poisoned destination; it
    // proves the zeroing rule only if the Host compiler really resolves each
    // geometry to the writer it names.
    for (what, case, expected) in overwrite_cases() {
        let resolution = Context::<f64>::default()
            .compile_storage_contract_resolution(
                engine,
                &case.dst(),
                FusionOperand::direct(case.lhs.space()),
                FusionOperand::direct(case.rhs.space()),
                case.axes(),
            )
            .unwrap();
        assert_eq!(
            resolution.direct_destination_inactive_blocks(),
            expected,
            "{what}"
        );
        assert_eq!(
            what.starts_with("core route"),
            !resolution.is_dynamic_tree(),
            "{what}"
        );
    }
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;

    use num_complex::Complex64;
    use symtn_dense::{CudaDenseContext, CudaScalar};
    use symtn_operations::cuda::CudaStorage;
    use symtn_operations::CudaTreeTransformExecutor;

    use crate::contract::resolution::{ContractRoute, StorageContractResolution};
    use crate::CudaContractScratch;

    trait Payload:
        CudaScalar + DenseRecouplingScalar + RecouplingCoefficientAction<f64> + std::fmt::Debug
    {
        fn fill(index: usize) -> Self;
        fn nan() -> Self;
        fn distance(self, other: Self) -> f64;
        fn magnitude(self) -> f64;
    }

    impl Payload for f64 {
        fn fill(index: usize) -> Self {
            (index as f64 * 0.37 + 0.1).sin()
        }
        fn nan() -> Self {
            f64::NAN
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).abs()
        }
        fn magnitude(self) -> f64 {
            self.abs()
        }
    }

    impl Payload for Complex64 {
        fn fill(index: usize) -> Self {
            Complex64::new(
                (index as f64 * 0.37 + 0.1).sin(),
                (index as f64 * 0.23).cos(),
            )
        }
        fn nan() -> Self {
            Complex64::new(f64::NAN, f64::NAN)
        }
        fn distance(self, other: Self) -> f64 {
            (self - other).norm()
        }
        fn magnitude(self) -> f64 {
            self.norm()
        }
    }

    fn data<D: Payload>(space: &crate::DynamicFusionMapSpace, salt: usize) -> Vec<D> {
        (0..space.required_len().unwrap())
            .map(|index| D::fill(index * 7 + salt))
            .collect()
    }

    fn assert_close<D: Payload>(actual: &[D], expected: &[D], what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: length");
        assert!(
            expected.iter().any(|value| value.magnitude() > 1e-3),
            "{what}: vacuous all-zero reference"
        );
        for (index, (&left, &right)) in actual.iter().zip(expected).enumerate() {
            assert!(
                left.distance(right) <= 1e-12 * (1.0 + right.magnitude()),
                "{what}: element {index} is {left:?}, expected {right:?}"
            );
        }
    }

    struct Replay<D: Payload> {
        engine: symtn_dense::CudaEngine,
        transforms: CudaTreeTransformExecutor,
        scratch: CudaContractScratch,
        _payload: std::marker::PhantomData<D>,
    }

    impl<D: Payload> Replay<D> {
        fn new() -> Self {
            Self {
                engine: symtn_dense::CudaEngine::open(symtn_dense::CudaDeviceId::from_ordinal(0))
                    .expect("a CUDA device"),
                transforms: CudaTreeTransformExecutor::default(),
                scratch: CudaContractScratch::default(),
                _payload: std::marker::PhantomData,
            }
        }

        /// Device and Host replays of one artifact for one forced candidate.
        fn run<R>(
            &mut self,
            case: &Case<R>,
            candidate: &crate::contract::fusion::ContractAxisOrderCandidate,
            orientation: FusionContractOrientation,
        ) -> (Vec<D>, Vec<D>, bool, bool)
        where
            R: MultiplicityFreeRigidSymbols<Scalar = f64>
                + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        {
            let (device, host, borrowed, _) = self.run_twisted(case, candidate, orientation);
            (device, host, borrowed.0, borrowed.1)
        }

        /// [`Self::run`], also reporting whether the artifact twisted.
        fn run_twisted<R>(
            &mut self,
            case: &Case<R>,
            candidate: &crate::contract::fusion::ContractAxisOrderCandidate,
            orientation: FusionContractOrientation,
        ) -> (Vec<D>, Vec<D>, (bool, bool), bool)
        where
            R: MultiplicityFreeRigidSymbols<Scalar = f64>
                + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        {
            let dst = case.dst();
            let lhs = data::<D>(case.lhs.space(), 1);
            let rhs = data::<D>(case.rhs.space(), 2);
            let (artifact, host, _) = forced_artifact(case, candidate, orientation, &lhs, &rhs);
            let borrowed = artifact.borrowed_sources();
            let twisted = artifact.requires_source_twist();
            let resolution =
                StorageContractResolution::new(ContractRoute::DynamicTree(Arc::new(artifact)));
            let device = self.execute(
                &resolution,
                &dst,
                (case.lhs.space().structure(), &lhs),
                (case.rhs.space().structure(), &rhs),
            );
            (device, host, borrowed, twisted)
        }

        /// Replays `resolution` into a fresh zero destination and into a
        /// retained NaN-poisoned one (`dst_is_zeroed = false`, the
        /// `contract_into` mode), asserts the two agree, and
        /// returns the first.
        fn execute<R>(
            &mut self,
            resolution: &StorageContractResolution<f64>,
            dst: &BoundDynamicFusionMapSpace<R>,
            (lhs_structure, lhs): (&Arc<symtn_core::BlockStructure>, &[D]),
            (rhs_structure, rhs): (&Arc<symtn_core::BlockStructure>, &[D]),
        ) -> Vec<D> {
            let ctx = &mut CudaDenseContext::new(&mut self.engine).unwrap();
            let lhs = CudaStorage::<D>::upload(ctx, lhs).unwrap();
            let rhs = CudaStorage::<D>::upload(ctx, rhs).unwrap();
            let len = dst.space().required_len().unwrap();
            let mut results = Vec::new();
            for (fill, init) in [
                (D::ZERO, crate::ContractDestinationInit::Zeroed),
                (D::nan(), crate::ContractDestinationInit::Axpby(D::ZERO)),
            ] {
                let mut out = CudaStorage::<D>::upload_owned(ctx, vec![fill; len]).unwrap();
                crate::execute_storage_contract_resolution_on_cuda(
                    ctx,
                    &mut self.transforms,
                    &mut self.scratch,
                    resolution,
                    (dst.space().structure(), &mut out),
                    (lhs_structure, &lhs),
                    (rhs_structure, &rhs),
                    D::ONE,
                    init,
                )
                .unwrap();
                results.push(out.download(ctx).unwrap());
            }
            let overwritten = results.pop().unwrap();
            let zeroed = results.pop().unwrap();
            assert_eq!(overwritten.len(), zeroed.len());
            for (index, (&left, &right)) in overwritten.iter().zip(&zeroed).enumerate() {
                assert!(
                    left.distance(right) <= 1e-12 * (1.0 + right.magnitude()),
                    "retained destination element {index} is {left:?}, fresh {right:?}"
                );
            }
            zeroed
        }
    }

    fn every_candidate_and_orientation<R, D>(case: &Case<R>, what: &str) -> (bool, bool)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        D: Payload,
    {
        let mut replay = Replay::<D>::new();
        let mut any_borrow = (false, false);
        for candidate in
            crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
        {
            for orientation in [
                FusionContractOrientation::LhsRhs,
                FusionContractOrientation::RhsLhs,
            ] {
                let (device, host, lhs_borrowed, rhs_borrowed) =
                    replay.run(case, &candidate, orientation);
                any_borrow.0 |= lhs_borrowed;
                any_borrow.1 |= rhs_borrowed;
                assert_close(
                    &device,
                    &host,
                    &format!("{what} {candidate:?} {orientation:?}"),
                );
            }
        }
        any_borrow
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn every_forced_candidate_and_orientation_matches_the_host_replay() {
        // What: both orientations of every axis-order candidate, U(1) rank 5
        // and SU(2) recoupling, real and complex, device == Host of the same
        // artifact.
        every_candidate_and_orientation::<_, f64>(&u1_case(), "U(1) f64");
        every_candidate_and_orientation::<_, Complex64>(&u1_case(), "U(1) c64");
        every_candidate_and_orientation::<_, f64>(&su2_case(), "SU(2) f64");
        every_candidate_and_orientation::<_, Complex64>(&su2_case(), "SU(2) c64");
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn an_identity_source_transform_borrows_either_operand_on_device() {
        // `lhs` already has its contracted legs as its whole domain in
        // order, `rhs` its as its whole codomain: under LhsRhs the lhs is
        // borrowed, and the permuted output keeps an output transform.
        let lhs_canonical = lhs_identity_case();
        let borrowed = every_candidate_and_orientation::<_, f64>(&lhs_canonical, "lhs borrow");
        let rhs_canonical = rhs_identity_case();
        let rhs_borrowed = every_candidate_and_orientation::<_, f64>(&rhs_canonical, "rhs borrow");
        // What: non-vacuity — some forced artifact really borrowed each side.
        assert!(borrowed.0, "no artifact borrowed the lhs");
        assert!(rhs_borrowed.1, "no artifact borrowed the rhs");
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn a_nan_poisoned_core_destination_scratch_is_zeroed_on_its_inactive_blocks() {
        // `v` carries charge 2 that `w` lacks: the core destination has a
        // (2, 2)-coupled block no GEMM job writes, so a retained scratch that
        // still holds NaN leaks unless exactly those blocks are zeroed.
        let provider = Arc::new(U1FusionRule);
        let v = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(0).sector_id(), 1),
                    (U1Irrep::new(1).sector_id(), 2),
                    (U1Irrep::new(2).sector_id(), 1),
                ],
                false,
            )
        };
        let w = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(0).sector_id(), 2),
                    (U1Irrep::new(1).sector_id(), 1),
                ],
                false,
            )
        };
        let case = Case {
            lhs: space(&provider, vec![v(), v()], vec![w()]),
            rhs: space(&provider, vec![w()], vec![v()]),
            lhs_axes: vec![2],
            rhs_axes: vec![0],
            output_axes: vec![1, 0, 2],
        };
        let mut replay = Replay::<f64>::new();
        let candidate = crate::contract::contracted_axis_order_candidates(&[2], &[0]).remove(0);
        let (first, host, _, _) = replay.run(&case, &candidate, FusionContractOrientation::LhsRhs);
        assert_close(&first, &host, "cold");
        let poisoned = replay.scratch.poison_core_destination::<f64>(
            &CudaDenseContext::new(&mut replay.engine).unwrap(),
            f64::nan(),
        );
        assert!(poisoned > 0, "the first replay retained a core destination");
        let (second, host, _, _) = replay.run(&case, &candidate, FusionContractOrientation::LhsRhs);
        assert!(second.iter().all(|value| value.is_finite()), "{second:?}");
        assert_close(&second, &host, "after poisoning");
    }

    fn every_fermionic_candidate_and_orientation<R, D>(case: &Case<R>, what: &str)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
        D: Payload,
    {
        let mut replay = Replay::<D>::new();
        let mut twisted = [false; 2];
        for candidate in
            crate::contract::contracted_axis_order_candidates(&case.lhs_axes, &case.rhs_axes)
        {
            for (slot, orientation) in orientations().into_iter().enumerate() {
                let (device, host, _, twist) = replay.run_twisted(case, &candidate, orientation);
                twisted[slot] |= twist;
                assert_close(
                    &device,
                    &host,
                    &format!("{what} {candidate:?} {orientation:?}"),
                );
            }
        }
        assert!(
            twisted.iter().any(|&twist| twist),
            "{what}: nothing twisted"
        );
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn a_twisted_artifact_replays_with_destination_scales_on_device_and_host_alike() {
        // What: for every forced candidate and both orientations — the
        // twisted operand is whichever one the artifact already materializes
        // — the device's θ-scaled source transform equals the Host's, both
        // folding the same destination scales into their moves, on the
        // same artifact; fZ2 x U(1) and fZ2 (x) SU(2) (recoupling), the twist
        // on one or both contracted legs, and the canonical non-uniform form.
        for (what, case) in fermionic_cases() {
            every_fermionic_candidate_and_orientation::<_, f64>(&case, what);
            every_fermionic_candidate_and_orientation::<_, Complex64>(&case, what);
        }
        for (what, case) in fermionic_su2_cases() {
            every_fermionic_candidate_and_orientation::<_, f64>(&case, what);
            every_fermionic_candidate_and_orientation::<_, Complex64>(&case, what);
        }
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn a_retained_destination_is_completed_by_every_destination_writer() {
        // What: into a NaN-poisoned destination the core route and an
        // identity output zero exactly the plan's inactive blocks, a fully
        // covered one needs none, and an output transform writes everything
        // itself — each equal to the eager Host contraction.
        fn at<D: Payload>() {
            let mut replay = Replay::<D>::new();
            for (what, case, _) in overwrite_cases() {
                let dst = case.dst();
                let lhs = data::<D>(case.lhs.space(), 1);
                let rhs = data::<D>(case.rhs.space(), 2);
                let resolution = Context::<D>::default()
                    .compile_storage_contract_resolution(
                        &mut replay.engine,
                        &dst,
                        FusionOperand::direct(case.lhs.space()),
                        FusionOperand::direct(case.rhs.space()),
                        case.axes(),
                    )
                    .unwrap();
                let device = replay.execute(
                    &resolution,
                    &dst,
                    (case.lhs.space().structure(), &lhs),
                    (case.rhs.space().structure(), &rhs),
                );
                assert_close(&device, &eager_host(&case, &lhs, &rhs), what);
            }
        }
        at::<f64>();
        at::<Complex64>();
    }
}

/// One route kind of the planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RouteKind {
    Core,
    CopyC,
    DynamicTree,
}

fn planner_route<R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    (_, lhs): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>),
    (_, rhs): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>),
    (lhs_axes, rhs_axes, output): (&[usize], &[usize], &[usize]),
) -> RouteKind
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::tree_transform::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let resolution = Context::<f64>::default()
        .plan_contract::<crate::DirectCoreExecutor, R>(
            engine, dst, lhs, rhs, lhs_axes, rhs_axes, output,
        )
        .unwrap();
    if resolution.copy_c().is_some() {
        RouteKind::CopyC
    } else if resolution.is_dynamic_tree() {
        RouteKind::DynamicTree
    } else {
        RouteKind::Core
    }
}

/// The route the Host eager entry takes, observed on a real execution.
fn host_eager_route<R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    (_, lhs): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>),
    (_, rhs): (&BoundDynamicFusionMapSpace<R>, FusionOperand<'_>),
    (lhs_axes, rhs_axes, output): (&[usize], &[usize], &[usize]),
) -> RouteKind
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::tree_transform::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let lhs_data = host_data(lhs.storage_space(), 3);
    let rhs_data = host_data(rhs.storage_space(), 4);
    let mut out = vec![0.0; dst.space().required_len().unwrap()];
    let mut context = Context::<f64>::default();
    context
        .tensorcontract_prelowered_into(
            engine,
            dst,
            &mut out,
            lhs,
            &lhs_data,
            rhs,
            &rhs_data,
            TensorContractSpec::new_with_conjugation(
                lhs_axes,
                rhs_axes,
                OutputAxisOrder::from_axes(output),
                lhs.storage_conjugate(),
                rhs.storage_conjugate(),
            ),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();
    if context.last_resolution_is_core() {
        RouteKind::Core
    } else if context.last_resolution_orientation().is_some() {
        RouteKind::DynamicTree
    } else {
        RouteKind::CopyC
    }
}

fn owned_routes<R>(case: &Case<R>) -> (RouteKind, RouteKind)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::tree_transform::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let dst = case.dst();
    let lhs = (&case.lhs, FusionOperand::direct(case.lhs.space()));
    let rhs = (&case.rhs, FusionOperand::direct(case.rhs.space()));
    let axes = (
        &case.lhs_axes[..],
        &case.rhs_axes[..],
        &case.output_axes[..],
    );
    (
        host_eager_route(&dst, lhs, rhs, axes),
        planner_route(&dst, lhs, rhs, axes),
    )
}

/// `lhs(v,v ← v,v) · rhs(v,v ← v,v)` on the given axes and output: the
/// TensorKit `copyC` probes C1p/C2p and the zero-copy controls C1/C2.
fn square_case<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    leg: impl Fn() -> SectorLeg,
    lhs_axes: [usize; 2],
    rhs_axes: [usize; 2],
    output_axes: [usize; 4],
) -> Case<R> {
    Case {
        lhs: space(provider, vec![leg(), leg()], vec![leg(), leg()]),
        rhs: space(provider, vec![leg(), leg()], vec![leg(), leg()]),
        lhs_axes: lhs_axes.to_vec(),
        rhs_axes: rhs_axes.to_vec(),
        output_axes: output_axes.to_vec(),
    }
}

/// #1858: Host eager and the device planner take the same route on every
/// fixture, owned and lazy, Core, CopyC and DynamicTree, bosonic and
/// fermionic. A fermionic twist uniform within each RHS coupled-sector matrix
/// rides the core as per-job GEMM alpha rather than a DynamicTree copy and
/// twist of an operand, as TensorKit's `blas_contract!`
/// (`tensoroperations.jl:399-429` @cfaa073) does.
#[test]
fn host_eager_and_the_device_planner_take_the_same_route() {
    let u1 = Arc::new(U1FusionRule);
    let su2 = Arc::new(SU2FusionRule);
    let u1_v = || u1_leg(false);
    let bosonic: Vec<(&str, (RouteKind, RouteKind), RouteKind)> = vec![
        (
            "U1 rank 5",
            owned_routes(&u1_case()),
            RouteKind::DynamicTree,
        ),
        ("SU2", owned_routes(&su2_case()), RouteKind::DynamicTree),
        (
            "SU2 rank 5",
            owned_routes(&su2_rank5_case()),
            RouteKind::DynamicTree,
        ),
        (
            "U1 lhs identity",
            owned_routes(&lhs_identity_case()),
            RouteKind::DynamicTree,
        ),
        (
            "U1 rhs identity",
            owned_routes(&rhs_identity_case()),
            RouteKind::DynamicTree,
        ),
        (
            "U1 C0 (mul!)",
            owned_routes(&square_case(&u1, u1_v, [2, 3], [0, 1], [0, 1, 2, 3])),
            RouteKind::Core,
        ),
        (
            "U1 C1 (sorted)",
            owned_routes(&square_case(&u1, u1_v, [3, 2], [1, 0], [0, 1, 2, 3])),
            RouteKind::Core,
        ),
        (
            "U1 C2 (swapped)",
            owned_routes(&square_case(&u1, u1_v, [0, 1], [2, 3], [2, 3, 0, 1])),
            RouteKind::Core,
        ),
        (
            "U1 C1p",
            owned_routes(&square_case(&u1, u1_v, [3, 2], [1, 0], [1, 0, 3, 2])),
            RouteKind::CopyC,
        ),
        (
            "U1 C2p",
            owned_routes(&square_case(&u1, u1_v, [0, 1], [2, 3], [3, 2, 1, 0])),
            RouteKind::CopyC,
        ),
        (
            "SU2 C1p",
            owned_routes(&square_case(&su2, su2_leg, [3, 2], [1, 0], [1, 0, 3, 2])),
            RouteKind::CopyC,
        ),
    ];
    for (name, (host, planner), expected) in bosonic {
        assert_eq!((host, planner), (expected, expected), "{name}");
    }

    // Lazy adjoints: the canonical lazy core, a lazy CopyC (L3p), a lazy
    // DynamicTree, and the two self-dual core-form lazy contractions the
    // Host resolves to `Structure`.
    let x = space(&su2, vec![su2_leg(), su2_leg()], vec![su2_leg(), su2_leg()]);
    let x_adjoint = x.adjoint_view().unwrap();
    let rhs = space(&su2, vec![su2_leg(), su2_leg()], vec![su2_leg()]);
    let lhs = space(&su2, vec![su2_leg()], vec![su2_leg(), su2_leg()]);
    let y = space(&su2, vec![su2_leg()], vec![su2_leg(), su2_leg()]);
    let y_adjoint = y.adjoint_view().unwrap();
    let square = space(&su2, vec![su2_leg(), su2_leg()], vec![su2_leg(), su2_leg()]);
    let p = space(&su2, vec![su2_leg(), su2_leg()], vec![su2_leg(), su2_leg()]);
    let p_adjoint = p.adjoint_view().unwrap();
    let u1_lhs = space(&u1, vec![u1_v(), u1_v()], vec![u1_v()]);
    let u1_lhs_adjoint = u1_lhs.adjoint_view().unwrap();
    let u1_rhs = space(&u1, vec![u1_v(), u1_v()], vec![u1_v()]);
    type Lazy<'a> = (
        &'a str,
        &'a BoundDynamicFusionMapSpace<SU2FusionRule>,
        FusionOperand<'a>,
        &'a BoundDynamicFusionMapSpace<SU2FusionRule>,
        FusionOperand<'a>,
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
        RouteKind,
    );
    // `symtn/tests/contract_cases::su2_core_form_lazy_cases` take `copyC`,
    // the predicate preceding the Host ladder; contracting a codomain and a
    // domain leg together leaves no zero-copy order, so DynamicTree.
    let lazy: Vec<Lazy<'_>> = vec![
        (
            "SU2 core-form lazy lhs",
            &x_adjoint,
            FusionOperand::adjoint(x.space()),
            &rhs,
            FusionOperand::direct(rhs.space()),
            vec![2, 3],
            vec![0, 1],
            vec![2, 0, 1],
            RouteKind::CopyC,
        ),
        (
            "SU2 core-form lazy rhs",
            &lhs,
            FusionOperand::direct(lhs.space()),
            &y_adjoint,
            FusionOperand::adjoint(y.space()),
            vec![1, 2],
            vec![0, 1],
            vec![1, 0],
            RouteKind::CopyC,
        ),
        (
            "SU2 lazy DynamicTree (lhs, mixed sides)",
            &x_adjoint,
            FusionOperand::adjoint(x.space()),
            &rhs,
            FusionOperand::direct(rhs.space()),
            vec![1, 3],
            vec![2, 0],
            vec![0, 1, 2],
            RouteKind::DynamicTree,
        ),
        (
            "SU2 lazy DynamicTree (rhs, mixed sides)",
            &square,
            FusionOperand::direct(square.space()),
            &p_adjoint,
            FusionOperand::adjoint(p.space()),
            vec![0, 2],
            vec![2, 0],
            vec![0, 1, 2, 3],
            RouteKind::DynamicTree,
        ),
        (
            "SU2 L4 (lazy mul!)",
            &p_adjoint,
            FusionOperand::adjoint(p.space()),
            &square,
            FusionOperand::direct(square.space()),
            vec![2, 3],
            vec![0, 1],
            vec![0, 1, 2, 3],
            RouteKind::Core,
        ),
        (
            "SU2 L3p (lazy copyC)",
            &p_adjoint,
            FusionOperand::adjoint(p.space()),
            &square,
            FusionOperand::direct(square.space()),
            vec![3, 2],
            vec![1, 0],
            vec![1, 0, 2, 3],
            RouteKind::CopyC,
        ),
    ];
    for (name, lhs_authority, lhs, rhs_authority, rhs, lhs_axes, rhs_axes, output, expected) in lazy
    {
        let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_oriented(
            lhs_authority,
            lhs,
            rhs_authority,
            rhs,
            &lhs_axes,
            &rhs_axes,
            OutputAxisOrder::from_axes(&output),
            None,
        )
        .unwrap();
        let operands = ((lhs_authority, lhs), (rhs_authority, rhs));
        let axes = (&lhs_axes[..], &rhs_axes[..], &output[..]);
        let host = host_eager_route(&dst, operands.0, operands.1, axes);
        let planner = planner_route(&dst, operands.0, operands.1, axes);
        assert_eq!((host, planner), (expected, expected), "{name}");
    }
    let output = [3, 0, 1, 2];
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_oriented(
        &u1_lhs_adjoint,
        FusionOperand::adjoint(u1_lhs.space()),
        &u1_rhs,
        FusionOperand::direct(u1_rhs.space()),
        &[1],
        &[0],
        OutputAxisOrder::from_axes(&output),
        None,
    )
    .unwrap();
    let operands = (
        (&u1_lhs_adjoint, FusionOperand::adjoint(u1_lhs.space())),
        (&u1_rhs, FusionOperand::direct(u1_rhs.space())),
    );
    let axes = (&[1][..], &[0][..], &output[..]);
    assert_eq!(
        (
            host_eager_route(&dst, operands.0, operands.1, axes),
            planner_route(&dst, operands.0, operands.1, axes),
        ),
        (RouteKind::DynamicTree, RouteKind::DynamicTree),
        "U1 lazy DynamicTree"
    );

    // Fermionic: the same route, and the uniform-twist `mul!` form (θ the
    // parity of the coupled sector) takes the scaled core.
    for (name, case) in fermionic_cases() {
        let (host, planner) = owned_routes(&case);
        assert_eq!(host, planner, "{name}");
    }
    for (name, case) in fermionic_su2_cases() {
        let (host, planner) = owned_routes(&case);
        assert_eq!(host, planner, "{name}");
    }
    let fu1 = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let fsu2 = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let uniform = [
        (
            "fZ2xU1 uniform mul!",
            owned_routes(&Case {
                lhs: space(
                    &fu1,
                    vec![fermion_u1_leg(&fu1, false)],
                    vec![fermion_u1_leg(&fu1, true)],
                ),
                rhs: space(
                    &fu1,
                    vec![fermion_u1_leg(&fu1, true)],
                    vec![fermion_u1_leg(&fu1, false)],
                ),
                lhs_axes: vec![1],
                rhs_axes: vec![0],
                output_axes: vec![0, 1],
            }),
        ),
        (
            "fZ2xSU2 uniform mul!",
            owned_routes(&Case {
                lhs: space(
                    &fsu2,
                    vec![fermion_su2_leg(&fsu2, false)],
                    vec![fermion_su2_leg(&fsu2, true)],
                ),
                rhs: space(
                    &fsu2,
                    vec![fermion_su2_leg(&fsu2, true)],
                    vec![fermion_su2_leg(&fsu2, false)],
                ),
                lhs_axes: vec![1],
                rhs_axes: vec![0],
                output_axes: vec![0, 1],
            }),
        ),
    ];
    for (name, routes) in uniform {
        assert_eq!(routes, (RouteKind::Core, RouteKind::Core), "{name}");
    }
}

/// #1858: the planner walks the requested order's zero-copy candidates once.
/// Before, `copyC`'s predicate walked it again after the core rung missed.
#[test]
fn the_planner_walks_the_requested_order_once() {
    let u1 = Arc::new(U1FusionRule);
    let u1_v = || u1_leg(false);
    let walks = |case: &Case<U1FusionRule>| {
        let dst = case.dst();
        let walks_rec = crate::probe::record();
        let route = planner_route(
            &dst,
            (&case.lhs, FusionOperand::direct(case.lhs.space())),
            (&case.rhs, FusionOperand::direct(case.rhs.space())),
            (&case.lhs_axes, &case.rhs_axes, &case.output_axes),
        );
        (route, walks_rec.get(crate::probe::Event::CandidateWalk))
    };
    // Core: the requested order only.
    assert_eq!(
        walks(&square_case(&u1, u1_v, [3, 2], [1, 0], [0, 1, 2, 3])),
        (RouteKind::Core, 1)
    );
    // CopyC: the requested order, the A·B default order (found), the
    // temporary's own core.
    assert_eq!(
        walks(&square_case(&u1, u1_v, [3, 2], [1, 0], [1, 0, 3, 2])),
        (RouteKind::CopyC, 3)
    );
    // DynamicTree: the requested order and both default orders.
    assert_eq!(walks(&u1_case()), (RouteKind::DynamicTree, 3));
}

/// #1488: `dim(C)` below both operands takes `copyC`. On the S1 sort
/// `A[3,2]·B[1,0] → [2,3,0,1]` and the S2 swap `A[0,1]·B[2,3] → [3,2,1,0]`
/// of `symtn/tests/contract_candidate_core.rs`
/// (`small_output_copy_c_shapes_match_tensorkit_with_one_output_transform`), with `v = u1{-1:8, 0:8,
/// 1:8}` and `c = u1{0:2, 1:1}`, the Host eager entry and the planner both
/// resolve to CopyC, and its core reads both operands untransformed: only
/// C is copied, never an operand. Whether the scorer ran is not observed.
#[test]
fn small_output_takes_copy_c_without_copying_an_operand() {
    let u1 = Arc::new(U1FusionRule);
    let leg = |entries: &[(i32, usize)]| {
        SectorLeg::new(
            entries
                .iter()
                .map(|&(charge, degeneracy)| (U1Irrep::new(charge).sector_id(), degeneracy)),
            false,
        )
    };
    let v = || leg(&[(-1, 8), (0, 8), (1, 8)]);
    let c = || leg(&[(0, 2), (1, 1)]);
    let cases = [
        (
            "S1",
            Case {
                lhs: space(&u1, vec![c(), c()], vec![v(), v()]),
                rhs: space(&u1, vec![v(), v()], vec![c(), c()]),
                lhs_axes: vec![3, 2],
                rhs_axes: vec![1, 0],
                output_axes: vec![2, 3, 0, 1],
            },
        ),
        (
            "S2",
            Case {
                lhs: space(&u1, vec![v(), v()], vec![c(), c()]),
                rhs: space(&u1, vec![c(), c()], vec![v(), v()]),
                lhs_axes: vec![0, 1],
                rhs_axes: vec![2, 3],
                output_axes: vec![3, 2, 1, 0],
            },
        ),
    ];
    for (name, case) in &cases {
        assert_eq!(
            owned_routes(case),
            (RouteKind::CopyC, RouteKind::CopyC),
            "{name}"
        );
        let engine = &mut symtn_dense::CpuEngine::serial();
        let resolution = Context::<f64>::default()
            .plan_contract::<crate::DirectCoreExecutor, _>(
                engine,
                &case.dst(),
                FusionOperand::direct(case.lhs.space()),
                FusionOperand::direct(case.rhs.space()),
                &case.lhs_axes,
                &case.rhs_axes,
                &case.output_axes,
            )
            .unwrap();
        let (core, _) = resolution.direct_core().unwrap();
        core.require_identity_direct_replay()
            .unwrap_or_else(|error| panic!("{name}: the core transforms an operand: {error}"));
    }
}

#[path = "route_host_tests.rs"]
mod route_host_tests;
