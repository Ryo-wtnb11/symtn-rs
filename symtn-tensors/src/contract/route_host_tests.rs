//! The DynamicTree route has one Host executor for eager and
//! member (any B) replays.
//!
//! Eager accumulation (`alpha`, `beta`, `Zeroed`, signed zeros, NaN) is
//! checked by hand formula in `route_host_core_tests.rs`; member replays
//! here against the dense physical-basis contraction of each member's own
//! operands, or, for the twisted fermionic fixture (no physical basis), the
//! eager contraction of those operands, whose values the typed layer checks
//! against TensorKit's `blas_contract!` steps on the same geometry
//! (`symtn/tests/engine_suites/contract.rs`, `contract_cases::fermionic_general`).

use super::*;
use crate::contract::fusion_block::FusionBlockContractWorkspace;
use symtn_operations::ContractDestinationInit;

/// Exactly representable payload in `{-2, -1, ±0, 1, 2}`.
fn exact_data(len: usize, salt: usize) -> Vec<f64> {
    (0..len)
        .map(|index| {
            let value = ((index * 7 + salt) % 5) as f64 - 2.0;
            if value == 0.0 && (index + salt) % 2 == 1 {
                -0.0
            } else {
                value
            }
        })
        .collect()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The DynamicTree fixtures: U(1) identity output with and without an
/// inactive core block, U(1) source and output transforms, and a twisted
/// fermionic contraction.
fn dynamic_tree_fixtures() -> (Vec<(&'static str, Case<U1FusionRule>)>, Case<FermionU1>) {
    let mut cases = overwrite_cases();
    cases.truncate(3);
    let u1 = cases
        .drain(1..)
        .map(|(name, case, _)| (name, case))
        .chain([("u1 source and output transforms", u1_case())])
        .map(|(name, case)| {
            assert_eq!(
                owned_routes(&case),
                (RouteKind::DynamicTree, RouteKind::DynamicTree),
                "{name}"
            );
            (name, case)
        })
        .collect::<Vec<_>>();
    let (_, fermionic) = fermionic_cases()
        .into_iter()
        .find(|(_, case)| {
            owned_routes(case) == (RouteKind::DynamicTree, RouteKind::DynamicTree)
                && planned_resolution(case).requires_source_twist()
        })
        .expect("a twisted fermionic DynamicTree fixture");
    (u1, fermionic)
}

fn planned_resolution<R>(case: &Case<R>) -> crate::contract::StorageContractResolution<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    Context::<f64>::default()
        .compile_storage_contract_resolution(
            engine,
            &case.dst(),
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap()
}

/// The Host member entry over default dense executors.
pub(super) fn run_members(
    artifact: &DynamicTreeExecutionArtifact<f64>,
    dst_structure: &Arc<symtn_core::BlockStructure>,
    workspace: &mut crate::contract::HostContractMembersWorkspace<f64>,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: &[f64],
    members: usize,
) -> Result<(), crate::OperationError> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    crate::contract::route_host::execute_dynamic_tree_execution_artifact_members_host(
        engine,
        &mut DenseTreeTransformOperations::default(),
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
        artifact,
        dst_structure,
        workspace,
        dst,
        lhs,
        rhs,
        members,
    )
}

fn close(found: f64, want: f64) -> bool {
    (found - want).abs() <= 1e-12 * (1.0 + want.abs())
}

/// Member replays of the planner's artifact at B = 1, 2, 1 on one workspace
/// into NaN-poisoned destinations; `check(member output, lhs, rhs, what)`
/// compares each member with the contraction of its own operands.
fn check_members<R>(name: &str, case: &Case<R>, mut check: impl FnMut(&[f64], &[f64], &[f64], &str))
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let resolution = planned_resolution(case);
    let crate::contract::resolution::ContractRoute::DynamicTree(artifact) = &resolution.route
    else {
        panic!("{name}: fixture must plan DynamicTree");
    };
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let mut workspace = crate::contract::HostContractMembersWorkspace::default();
    for (call, members) in [1, 2, 1].into_iter().enumerate() {
        let lhs = (0..members)
            .flat_map(|member| host_data(case.lhs.space(), 1 + 2 * member + call))
            .collect::<Vec<_>>();
        let rhs = (0..members)
            .flat_map(|member| host_data(case.rhs.space(), 5 + 3 * member + call))
            .collect::<Vec<_>>();
        let mut out = vec![f64::NAN; len * members];
        run_members(
            artifact,
            dst.space().structure(),
            &mut workspace,
            &mut out,
            &lhs,
            &rhs,
            members,
        )
        .unwrap();
        for member in 0..members {
            check(
                &out[member * len..(member + 1) * len],
                &lhs[member * lhs_len..(member + 1) * lhs_len],
                &rhs[member * rhs_len..(member + 1) * rhs_len],
                &format!("{name}, B = {members}, member {member}"),
            );
        }
    }
}

#[test]
fn dynamic_tree_member_replays_match_each_members_contraction() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: on one workspace across B = 1, 2, 1, each member of a
    // DynamicTree replay (identity output with and without an inactive
    // core block, source and output transforms, twisted fermionic sources)
    // equals the contraction of its own operands.
    let (u1, fermionic) = dynamic_tree_fixtures();
    let has_core_dst = |resolution: crate::contract::StorageContractResolution<f64>| {
        resolution.direct_destination_inactive_blocks().is_none()
    };
    assert_eq!(
        u1.iter()
            .map(|(_, case)| has_core_dst(planned_resolution(case)))
            .collect::<Vec<_>>(),
        [false, false, true]
    );
    for (name, case) in &u1 {
        let dst = case.dst();
        check_members(name, case, |member, lhs, rhs, what| {
            let (oracle_shape, oracle) = physical_oracle(case, lhs, rhs);
            let (shape, found) = crate::expand_physical_host(
                crate::BoundDynamicTensorRef::try_new(&dst, member).unwrap(),
            )
            .unwrap();
            assert_eq!(shape, oracle_shape, "{what}");
            assert!(oracle.iter().any(|&value| value != 0.0), "{what}");
            for (index, (&found, &want)) in found.iter().zip(&oracle).enumerate() {
                assert!(close(found, want), "{what} [{index}]: {found} vs {want}");
            }
        });
    }
    let len = fermionic.dst().space().required_len().unwrap();
    check_members("fermionic twisted", &fermionic, |member, lhs, rhs, what| {
        let mut eager = vec![0.0; len];
        Context::<f64>::default()
            .tensorcontract_prelowered_into(
                engine,
                &fermionic.dst(),
                &mut eager,
                FusionOperand::direct(fermionic.lhs.space()),
                lhs,
                FusionOperand::direct(fermionic.rhs.space()),
                rhs,
                fermionic.axes(),
                1.0,
                ContractDestinationInit::Zeroed,
            )
            .unwrap();
        assert!(eager.iter().any(|&value| value != 0.0), "{what}");
        for (index, (&found, &want)) in member.iter().zip(&eager).enumerate() {
            assert!(close(found, want), "{what} [{index}]: {found} vs {want}");
        }
    });
}

/// Runs `members` copies of `case` on `workspace` and returns each stage's
/// coefficient-pack conversions so far.
fn member_pack_builds<R>(
    case: &Case<R>,
    artifact: &DynamicTreeExecutionArtifact<f64>,
    workspace: &mut crate::contract::HostContractMembersWorkspace<f64>,
    members: usize,
) -> [usize; 3]
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let dst = case.dst();
    let lhs = host_data(case.lhs.space(), 3).repeat(members);
    let rhs = host_data(case.rhs.space(), 5).repeat(members);
    let mut out = vec![f64::NAN; members * dst.space().required_len().unwrap()];
    run_members(
        artifact,
        dst.space().structure(),
        workspace,
        &mut out,
        &lhs,
        &rhs,
        members,
    )
    .unwrap();
    assert!(out.iter().all(|value| value.is_finite()));
    workspace.coefficient_pack_builds()
}

#[test]
fn a_warm_one_member_replay_converts_no_coefficient_pack() {
    // Why: each member stage replays into its own workspace at every B,
    // so a warm B = 1 replay of Multi transforms finds all three packs
    // installed instead of rebuilding them per stage.
    let su2 = su2_rank5_case();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let v = || fermion_su2_leg(&provider, false);
    let fermionic = Case {
        lhs: space(&provider, vec![v(), v(), v()], vec![v(), v()]),
        rhs: space(&provider, vec![v(), v()], vec![v(), v()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    };
    fn check<R>(case: &Case<R>, what: &str)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
    {
        let resolution = planned_resolution(case);
        let crate::contract::resolution::ContractRoute::DynamicTree(artifact) = &resolution.route
        else {
            panic!("{what}: fixture must plan DynamicTree");
        };
        assert!(artifact.test_has_core_dst(), "{what}");
        assert_eq!(artifact.borrowed_sources(), (false, false), "{what}");
        let mut workspace = crate::contract::HostContractMembersWorkspace::default();
        let cold = member_pack_builds(case, artifact, &mut workspace, 1);
        assert!(cold.iter().any(|&builds| builds > 0), "{what}: {cold:?}");
        assert!(cold.iter().all(|&builds| builds <= 1), "{what}: {cold:?}");
        for _ in 0..3 {
            assert_eq!(
                member_pack_builds(case, artifact, &mut workspace, 1),
                cold,
                "{what}"
            );
        }
    }
    check(&su2, "SU2");
    check(&fermionic, "fZ2xSU2");
}

#[test]
fn a_warm_eager_contraction_converts_no_coefficient_pack() {
    // Why: the eager lhs, rhs and output transforms each replay into
    // their own workspace, so a warm call finds all three Multi coefficient
    // packs installed instead of evicting each other's. Isolated: an eager
    // call resolves its transformers through the process-global cache, which
    // a concurrent test's clear would rebuild under a fresh identity.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_WARM_EAGER_PACKS_ISOLATED",
        "contract::storage_contract_tests::route_host_tests::a_warm_eager_contraction_converts_no_coefficient_pack",
    ) {
        return;
    }
    fn check<R>(case: &Case<R>, what: &str)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
    {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let resolution = planned_resolution(case);
        let crate::contract::resolution::ContractRoute::DynamicTree(artifact) = &resolution.route
        else {
            panic!("{what}: fixture must plan DynamicTree");
        };
        assert!(artifact.test_has_core_dst(), "{what}");
        assert_eq!(artifact.borrowed_sources(), (false, false), "{what}");
        let dst = case.dst();
        let len = dst.space().required_len().unwrap();
        let lhs = exact_data(case.lhs.space().required_len().unwrap(), 1);
        let rhs = exact_data(case.rhs.space().required_len().unwrap(), 3);
        // Values are checked by oracle elsewhere; here every warm replay must
        // give the cold replay's bits.
        let expected = std::cell::RefCell::new(None::<Vec<u64>>);
        let mut run = |context: &mut Context<f64>| {
            let mut out = vec![f64::NAN; len];
            context
                .tensorcontract_prelowered_into(
                    engine,
                    &dst,
                    &mut out,
                    FusionOperand::direct(case.lhs.space()),
                    &lhs,
                    FusionOperand::direct(case.rhs.space()),
                    &rhs,
                    case.axes(),
                    1.0,
                    ContractDestinationInit::Axpby(0.0),
                )
                .unwrap();
            let mut expected = expected.borrow_mut();
            assert_eq!(
                bits(&out),
                *expected.get_or_insert_with(|| bits(&out)),
                "{what}"
            );
            context.eager_coefficient_pack_builds()
        };
        let mut context = Context::<f64>::default();
        let cold = run(&mut context);
        assert_eq!(cold, [1, 1, 1], "{what}: all three stages recouple");
        for _ in 0..3 {
            assert_eq!(run(&mut context), cold, "{what}");
        }
        // The stage workspaces are retained Host scratch: counted, and
        // released by trim (a fresh workspace has converted no pack).
        let stage_bytes = context.stage_workspace_retained_bytes();
        assert!(stage_bytes > 0, "{what}");
        assert!(
            context.retained_host_scratch_bytes() >= stage_bytes,
            "{what}"
        );
        context.trim_host_scratch();
        assert_eq!(context.retained_host_scratch_bytes(), 0, "{what}");
        assert_eq!(context.eager_coefficient_pack_builds(), [0; 3], "{what}");
        assert_eq!(run(&mut context), cold, "{what}");
    }
    check(&su2_rank5_case(), "SU2");
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let v = || fermion_su2_leg(&provider, false);
    check(
        &Case {
            lhs: space(&provider, vec![v(), v(), v()], vec![v(), v()]),
            rhs: space(&provider, vec![v(), v()], vec![v(), v()]),
            lhs_axes: vec![3, 1],
            rhs_axes: vec![0, 3],
            output_axes: vec![2, 0, 4, 1, 3],
        },
        "fZ2xSU2",
    );
}

#[test]
fn one_workspace_refills_its_core_destination_only_on_a_replay_change() {
    // Plan and member-count changes on one workspace, B = 1 included:
    // only a new replay (cold, B change, plan change) fills the inactive
    // core-destination blocks; values are checked per member each call.
    let mut cases = overwrite_cases();
    let (_, workspace_dst, _) = cases.pop().unwrap();
    let (_, caller_dst, _) = cases.remove(1);
    let workspace_artifact = members_artifact(&workspace_dst);
    let caller_artifact = members_artifact(&caller_dst);
    let mut ws = crate::contract::HostContractMembersWorkspace::default();
    let mut fills = |case, artifact, members, salt| {
        replay_members_checked(case, artifact, &mut ws, members, salt)
    };
    let a = (&workspace_dst, &*workspace_artifact);
    let b = (&caller_dst, &*caller_artifact);
    for ((case, artifact), members, expected) in [
        (a, 2, 1),
        (a, 17, 1),
        (a, 17, 0),
        (a, 1, 1),
        (a, 1, 0),
        (a, 2, 1),
        (b, 2, 1),
        (b, 1, 1),
        (a, 1, 1),
        (a, 1, 0),
    ] {
        assert_eq!(fills(case, artifact, members, 3 + members), expected);
    }
}

#[test]
fn a_failed_one_member_replay_rezeroes_on_retry() {
    let (_, case, _) = overwrite_cases().pop().unwrap();
    let artifact = members_artifact(&case);
    let mut ws = crate::contract::HostContractMembersWorkspace::default();
    assert_eq!(replay_members_checked(&case, &artifact, &mut ws, 1, 3), 1);
    assert_eq!(replay_members_checked(&case, &artifact, &mut ws, 1, 9), 0);
    let fresh = members_artifact(&case);
    ws.poison_dst_then_fail();
    assert!(try_replay_members_checked(&case, &fresh, &mut ws, 1, 5).is_err());
    assert_eq!(replay_members_checked(&case, &fresh, &mut ws, 1, 5), 1);
    assert_eq!(replay_members_checked(&case, &fresh, &mut ws, 1, 7), 0);
}

#[test]
fn member_replays_outside_the_overwrite_contract_are_unsupported_before_writes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    use crate::contract::route_host::{
        execute_dynamic_tree_route_host, EagerTreeStage, HostRouteScratch,
    };
    let (_, case, _) = overwrite_cases().pop().unwrap();
    let artifact = members_artifact(&case);
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs = host_data(case.lhs.space(), 3).repeat(2);
    let rhs = host_data(case.rhs.space(), 5).repeat(2);
    let mut tree_backend = DenseTreeTransformOperations::default();
    let mut tree_workspaces = <[symtn_operations::TreeTransformWorkspace<f64>; 3]>::default();
    let mut backend = DenseTreeTransformOperations::default();
    let mut backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
    let mut slot = crate::contract::route_host::CoreSlot::default();
    let (mut a, mut b, mut c) = Default::default();
    // Each row names the boundary that must reject it, so removing the
    // overwrite-contract guard fails the first two rows.
    const GUARD: &str = "member contraction overwrites its destination with unit alpha";
    for (alpha, init, with_slot, expected) in [
        (2.0, ContractDestinationInit::Axpby(0.0), true, GUARD),
        (1.0, ContractDestinationInit::Axpby(1.0), true, GUARD),
        // The eager stage replays exactly one member; `Zeroed` (a born-zero
        // destination) passes the overwrite guard and reaches it.
        (
            1.0,
            ContractDestinationInit::Axpby(0.0),
            true,
            "eager Host contraction replays exactly one member",
        ),
        (
            1.0,
            ContractDestinationInit::Zeroed,
            true,
            "eager Host contraction replays exactly one member",
        ),
        // A member replay needs a caller-owned core slot.
        (
            1.0,
            ContractDestinationInit::Axpby(0.0),
            false,
            "member contraction needs a caller-owned core slot",
        ),
    ] {
        let mut out = vec![f64::NAN; 2 * len];
        let error = execute_dynamic_tree_route_host(
            engine,
            &mut EagerTreeStage {
                backend: &mut tree_backend,
                workspaces: &mut tree_workspaces,
            },
            &mut crate::contract::fusion_block::BackendRank2Gemm::<_, _, f64>::new(
                &mut backend,
                &mut backend_workspace,
            ),
            &mut crate::contract::fusion_block::FusionBlockContractWorkspace::default(),
            HostRouteScratch {
                lhs: &mut a,
                rhs: &mut b,
                core_dst: &mut c,
                core: with_slot.then_some(&mut slot),
            },
            &artifact,
            (dst.space().structure(), &mut out),
            &lhs,
            &rhs,
            2,
            alpha,
            init,
        )
        .unwrap_err();
        assert!(
            matches!(
                error,
                crate::OperationError::UnsupportedTensorContractScope { message }
                    if message == expected
            ),
            "{error:?}"
        );
        assert!(out.iter().all(|value| value.is_nan()));
    }
}

#[path = "route_host_core_tests.rs"]
mod core_routes;
