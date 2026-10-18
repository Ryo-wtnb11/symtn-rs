//! The DynamicTree route has one Host executor for eager and
//! member (any B) replays (#1859).
//!
//! The characterization compares, bit for bit, the executor against `d416b1a2`'s
//! eager step sequence spelled with the replay primitives
//! ([`base_sequence`]), for every alpha/beta placement, `Zeroed`, signed-zero
//! and non-finite case. Both run the same dense kernels on the same machine,
//! so the comparison is exact on every platform. The pinned digests then tie
//! the values to `d416b1a2` across platforms: they fold signed zeros and NaNs
//! to one pattern each, because the sign of an exactly-zero GEMM sum and of a
//! default NaN are decided by the dense backend's architecture-specific
//! kernels (macOS AArch64 and Linux x86-64 differ), not by the executor.

use super::*;
use crate::contract::fusion_block::{BackendRank2Gemm, FusionBlockContractWorkspace};
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

/// The platform-independent part of a value's bits: one pattern for every
/// NaN and for both zeros.
fn portable_bits(value: f64) -> u64 {
    if value.is_nan() {
        f64::NAN.to_bits()
    } else if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn fnv(digest: &mut u64, values: &[f64]) {
    for &value in values {
        for byte in portable_bits(value).to_le_bytes() {
            *digest ^= u64::from(byte);
            *digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;

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
    Context::<f64>::default()
        .compile_storage_contract_resolution(
            &case.dst(),
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            case.axes(),
        )
        .unwrap()
}

fn inits() -> [(ContractDestinationInit<f64>, Option<f64>); 5] {
    // `None`: the destination starts NaN-poisoned (strong-zero beta) or zero
    // (`Zeroed`); `Some(fill)` marks a destination read through beta.
    [
        (ContractDestinationInit::Axpby(0.0), None),
        (ContractDestinationInit::Axpby(-0.0), Some(0.0)),
        (ContractDestinationInit::Axpby(1.0), Some(0.0)),
        (ContractDestinationInit::Axpby(2.0), Some(0.0)),
        (ContractDestinationInit::Zeroed, None),
    ]
}

/// `d416b1a2`'s eager DynamicTree step sequence, spelled with the replay
/// primitives and fresh resources: overwrite each owned source into zeroed
/// scratch (with its twist scales), `execute_raw(alpha, beta)` into the
/// destination for an identity output, else `execute_raw(alpha, 0)` into a
/// zeroed core destination and `into(1, beta)`; `beta = init.active_beta()`.
#[allow(clippy::too_many_arguments)]
fn base_sequence(
    artifact: &DynamicTreeExecutionArtifact<f64>,
    dst_structure: &Arc<symtn_core::BlockStructure>,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: &[f64],
    alpha: f64,
    init: ContractDestinationInit<f64>,
) {
    let beta = init.active_beta();
    let mut tree = TreeTransformExecutionContext::<f64, RuleIdentity>::new(
        DenseTreeTransformOperations::default(),
    );
    let [lhs_scales, rhs_scales] = artifact.stage_scales();
    let mut source = |space: &crate::DynamicFusionMapSpace,
                      structure: &crate::TreeTransformStructure<f64>,
                      replay: &Arc<symtn_core::BlockStructure>,
                      borrowed: bool,
                      data: &[f64],
                      scales: &[(usize, f64)]| {
        if borrowed {
            return data.to_vec();
        }
        let mut scratch = vec![0.0; space.required_len().unwrap()];
        tree.tree_transform_structure_overwrite_into_raw(
            structure,
            space.structure(),
            replay,
            &mut scratch,
            data,
            1.0,
            scales,
        )
        .unwrap();
        scratch
    };
    let (l, r) = (&artifact.lhs_transform, &artifact.rhs_transform);
    let lhs_core = source(
        &l.space,
        l.transform_structure.as_ref().unwrap(),
        &l.replay_structure,
        artifact.lhs_borrowed,
        lhs,
        lhs_scales,
    );
    let rhs_core = source(
        &r.space,
        r.transform_structure.as_ref().unwrap(),
        &r.replay_structure,
        artifact.rhs_borrowed,
        rhs,
        rhs_scales,
    );
    let (left, right) = artifact.core_order(&lhs_core[..], &rhs_core[..]);
    let (left_structure, right_structure) = artifact.core_order(
        artifact.lhs_transform.space.structure(),
        artifact.rhs_transform.space.structure(),
    );
    let mut backend = DenseTreeTransformOperations::default();
    let mut backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
    let mut gemm = BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut backend_workspace);
    let mut core = |out_structure: &Arc<symtn_core::BlockStructure>, out: &mut [f64], beta| {
        artifact
            .block_plan
            .execute_raw(
                &mut crate::StridedHostKernelAdapter::default(),
                &mut gemm,
                &mut FusionBlockContractWorkspace::default(),
                out_structure,
                out,
                left_structure,
                left,
                right_structure,
                right,
                alpha,
                beta,
            )
            .unwrap()
    };
    match &artifact.core_dst {
        None => core(dst_structure, dst, beta),
        Some(output) => {
            let mut core_dst = vec![0.0; output.space.required_len().unwrap()];
            core(output.space.structure(), &mut core_dst, 0.0);
            tree.tree_transform_structure_into_raw(
                &output.output_transform_structure,
                dst_structure,
                output.space.structure(),
                dst,
                &core_dst,
                1.0,
                beta,
            )
            .unwrap();
        }
    }
}

/// Every eager `(alpha, init)` replay plus non-finite operand replays: each
/// bit-identical to [`base_sequence`]; the portable digest of all outputs.
fn eager_digest<R>(case: &Case<R>) -> u64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs = exact_data(case.lhs.space().required_len().unwrap(), 1);
    let rhs = exact_data(case.rhs.space().required_len().unwrap(), 3);
    let resolution = Context::<f64>::default()
        .plan_contract::<crate::contract::HostEagerExecutor, _>(
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
        )
        .unwrap();
    let crate::contract::resolution::ContractRoute::DynamicTree(artifact) = &resolution.route
    else {
        panic!("eager must plan DynamicTree");
    };
    let mut digest = FNV_START;
    let mut context = Context::<f64>::default();
    let mut run = |lhs: &[f64], alpha: f64, init: ContractDestinationInit<f64>, out: Vec<f64>| {
        let mut expected = out.clone();
        base_sequence(
            artifact,
            dst.space().structure(),
            &mut expected,
            lhs,
            &rhs,
            alpha,
            init,
        );
        let mut actual = out;
        context
            .tensorcontract_prelowered_into(
                &dst,
                &mut actual,
                FusionOperand::direct(case.lhs.space()),
                lhs,
                FusionOperand::direct(case.rhs.space()),
                &rhs,
                case.axes(),
                alpha,
                init,
            )
            .unwrap();
        assert!(!context.last_resolution_is_core());
        assert!(context.last_resolution_orientation().is_some());
        assert_eq!(bits(&actual), bits(&expected), "alpha {alpha}, {init:?}");
        fnv(&mut digest, &actual);
    };
    for alpha in [0.0, 1.0, -1.0, 2.5] {
        for (init, read) in inits() {
            let out = match (init, read) {
                (ContractDestinationInit::Zeroed, _) => vec![0.0; len],
                (_, None) => vec![f64::NAN; len],
                (_, Some(_)) => exact_data(len, 11),
            };
            run(&lhs, alpha, init, out);
        }
    }
    let mut poisoned = lhs.clone();
    poisoned[0] = f64::NAN;
    if poisoned.len() > 1 {
        poisoned[1] = f64::INFINITY;
    }
    for alpha in [0.0, 1.0] {
        run(
            &poisoned,
            alpha,
            ContractDestinationInit::Axpby(1.0),
            exact_data(len, 11),
        );
    }
    digest
}

/// Member replays of the planner's artifact at B = 1, 2, 1 into NaN-poisoned
/// destinations: B = 1 bit-identical to [`base_sequence`] with unit alpha
/// and strong-zero beta, every member at B = 2 equal to it up to the
/// backend's zero and NaN signs (a batched GEMM may sign an exact zero
/// differently); the portable digest of all outputs.
fn member_digest<R>(case: &Case<R>) -> u64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let resolution = planned_resolution(case);
    let crate::contract::resolution::ContractRoute::DynamicTree(artifact) = &resolution.route
    else {
        panic!("fixture must plan DynamicTree");
    };
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let mut workspace = crate::contract::HostContractMembersWorkspace::default();
    let mut digest = FNV_START;
    for members in [1, 2, 1] {
        let lhs = exact_data(lhs_len * members, 1);
        let rhs = exact_data(rhs_len * members, 3);
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
            let mut expected = vec![f64::NAN; len];
            base_sequence(
                artifact,
                dst.space().structure(),
                &mut expected,
                &lhs[member * lhs_len..(member + 1) * lhs_len],
                &rhs[member * rhs_len..(member + 1) * rhs_len],
                1.0,
                ContractDestinationInit::Axpby(0.0),
            );
            let actual = &out[member * len..(member + 1) * len];
            if members == 1 {
                assert_eq!(bits(actual), bits(&expected), "B = 1");
            } else {
                let portable =
                    |values: &[f64]| values.iter().map(|&v| portable_bits(v)).collect::<Vec<_>>();
                assert_eq!(
                    portable(actual),
                    portable(&expected),
                    "B = {members}, member {member}"
                );
            }
        }
        fnv(&mut digest, &out);
    }
    digest
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
    crate::contract::route_host::execute_dynamic_tree_execution_artifact_members_host(
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

/// Portable digests (see [`portable_bits`]), identical for `d416b1a2` and this
/// executor, on macOS AArch64 and on Linux x86-64:
/// `(fixture, eager digest, member digest)`.
const PINNED: [(&str, u64, u64); 4] = [
    (
        "transformed lhs, identity output, inactive block",
        0x42d6_b575_aa34_ae94,
        0x04b5_09ec_131c_53cd,
    ),
    (
        "transformed rhs, identity output, fully covered",
        0x6203_f240_4ef7_5df5,
        0xdb99_a4e5_ec6b_0e4d,
    ),
    (
        "u1 source and output transforms",
        0x5a37_4453_eea1_39ed,
        0x3c43_b277_1a82_5538,
    ),
    (
        "fermionic twisted",
        0xe236_3de9_d940_c5e6,
        0xfe76_0e2e_41d4_0855,
    ),
];

#[test]
fn dynamic_tree_eager_and_member_bits_are_pinned() {
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
    let mut observed = u1
        .iter()
        .map(|(name, case)| (*name, eager_digest(case), member_digest(case)))
        .collect::<Vec<_>>();
    observed.push((
        "fermionic twisted",
        eager_digest(&fermionic),
        member_digest(&fermionic),
    ));
    assert_eq!(observed, PINNED, "observed digests: {observed:#x?}");
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
    let provider = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
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
    // Why (#2101): the eager lhs, rhs and output transforms each replay into
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
        let mut expected = vec![f64::NAN; len];
        base_sequence(
            artifact,
            dst.space().structure(),
            &mut expected,
            &lhs,
            &rhs,
            1.0,
            ContractDestinationInit::Axpby(0.0),
        );
        let run = |context: &mut Context<f64>| {
            let mut out = vec![f64::NAN; len];
            context
                .tensorcontract_prelowered_into(
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
            assert_eq!(bits(&out), bits(&expected), "{what}");
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
    let provider = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
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
        // destination, #2123) passes the overwrite guard and reaches it.
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
