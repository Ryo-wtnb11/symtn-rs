//! Core and CopyC routes on the one Host executor (#1859).
//!
//! The eager characterization compares, bit for bit and on the machine
//! running the test, every eager Core / CopyC replay with `e6e9cac8`'s eager
//! step sequence spelled with the replay primitives ([`base_core`],
//! [`base_copy_c`]): alpha/beta placement, `Zeroed`, signed zeros and
//! non-finite operands. The pinned digests fold signed zeros and NaNs (their
//! signs are chosen by the dense backend's architecture-specific kernels) and
//! tie the values to `e6e9cac8` on every platform.

use super::*;

/// A U(1) fixture of each Core / CopyC shape: unswapped Core with an
/// inactive block, swapped Core, and CopyC in both orientations.
fn core_fixtures() -> Vec<(&'static str, Case<U1FusionRule>, RouteKind)> {
    let u1 = Arc::new(U1FusionRule);
    let u1_v = || u1_leg(false);
    let (_, core, _) = overwrite_cases().remove(0);
    let fixtures = vec![
        ("core, inactive block", core, RouteKind::Core),
        (
            "core, swapped",
            square_case(&u1, u1_v, [0, 1], [2, 3], [2, 3, 0, 1]),
            RouteKind::Core,
        ),
        (
            "copyC C1p",
            square_case(&u1, u1_v, [3, 2], [1, 0], [1, 0, 3, 2]),
            RouteKind::CopyC,
        ),
        (
            "copyC C2p",
            square_case(&u1, u1_v, [0, 1], [2, 3], [3, 2, 1, 0]),
            RouteKind::CopyC,
        ),
    ];
    for (name, case, route) in &fixtures {
        assert_eq!(owned_routes(case), (*route, *route), "{name}");
    }
    fixtures
}

fn eager_resolution<R>(case: &Case<R>) -> crate::contract::StorageContractResolution<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    Context::<f64>::default()
        .plan_contract::<crate::contract::HostEagerExecutor, _>(
            &case.dst(),
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
        )
        .unwrap()
}

type Operand<'a> = (&'a Arc<symtn_core::BlockStructure>, &'a [f64]);

/// `e6e9cac8`'s eager Core arm: the operands swapped for the B·A candidate,
/// then `execute_raw_zeroed(alpha)` or `execute_raw(alpha, beta)`.
#[allow(clippy::too_many_arguments)]
fn base_core(
    plan: &symtn_operations::fusion_replay::FusionBlockContractPlan<f64>,
    swapped: bool,
    dst_structure: &Arc<symtn_core::BlockStructure>,
    dst: &mut [f64],
    lhs: Operand<'_>,
    rhs: Operand<'_>,
    alpha: f64,
    init: ContractDestinationInit<f64>,
) {
    let ((left_structure, left), (right_structure, right)) =
        if swapped { (rhs, lhs) } else { (lhs, rhs) };
    let mut backend = DenseTreeTransformOperations::default();
    let mut backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
    let mut gemm = BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut backend_workspace);
    let mut kernels = crate::StridedHostKernelAdapter::default();
    let mut workspace = FusionBlockContractWorkspace::default();
    match init {
        ContractDestinationInit::Zeroed => plan.execute_raw_zeroed(
            &mut kernels,
            &mut gemm,
            &mut workspace,
            dst_structure,
            dst,
            left_structure,
            left,
            right_structure,
            right,
            alpha,
        ),
        ContractDestinationInit::Axpby(beta) => plan.execute_raw(
            &mut kernels,
            &mut gemm,
            &mut workspace,
            dst_structure,
            dst,
            left_structure,
            left,
            right_structure,
            right,
            alpha,
            beta,
        ),
    }
    .unwrap();
}

/// `e6e9cac8`'s eager CopyC arm (TensorKit `blas_contract!` copyC): the core
/// `(1, 0)` into a zeroed temporary, then `overwrite(alpha)` for `Zeroed`
/// or `into(alpha, beta)`.
fn base_copy_c(
    copy: &crate::contract::CopyCRoute<f64>,
    dst_structure: &Arc<symtn_core::BlockStructure>,
    dst: &mut [f64],
    lhs: Operand<'_>,
    rhs: Operand<'_>,
    alpha: f64,
    init: ContractDestinationInit<f64>,
) {
    let mut temporary = vec![0.0; copy.temporary_len];
    base_core(
        &copy.core,
        copy.swapped,
        &copy.temporary,
        &mut temporary,
        lhs,
        rhs,
        1.0,
        ContractDestinationInit::Axpby(0.0),
    );
    let mut tree = TreeTransformExecutionContext::<f64, RuleIdentity>::new(
        DenseTreeTransformOperations::default(),
    );
    match init {
        ContractDestinationInit::Zeroed => tree.tree_transform_structure_overwrite_into_raw(
            &copy.transform,
            dst_structure,
            &copy.temporary,
            dst,
            &temporary,
            alpha,
            &[],
        ),
        ContractDestinationInit::Axpby(beta) => tree.tree_transform_structure_into_raw(
            &copy.transform,
            dst_structure,
            &copy.temporary,
            dst,
            &temporary,
            alpha,
            beta,
        ),
    }
    .unwrap();
}

/// `e6e9cac8`'s eager sequence of a Core or CopyC resolution.
#[allow(clippy::too_many_arguments)]
fn base_route(
    resolution: &crate::contract::StorageContractResolution<f64>,
    dst_structure: &Arc<symtn_core::BlockStructure>,
    dst: &mut [f64],
    lhs: Operand<'_>,
    rhs: Operand<'_>,
    alpha: f64,
    init: ContractDestinationInit<f64>,
) {
    match &resolution.route {
        crate::contract::resolution::ContractRoute::Core { plan, swapped } => {
            base_core(plan, *swapped, dst_structure, dst, lhs, rhs, alpha, init)
        }
        crate::contract::resolution::ContractRoute::CopyC(copy) => {
            base_copy_c(copy, dst_structure, dst, lhs, rhs, alpha, init)
        }
        crate::contract::resolution::ContractRoute::DynamicTree(_) => {
            panic!("Core / CopyC fixture planned DynamicTree")
        }
    }
}

/// Every eager `(alpha, init)` replay plus non-finite operand replays, each
/// bit-identical to [`base_route`]; the portable digest of all outputs.
fn eager_core_digest<R>(case: &Case<R>, route: RouteKind) -> u64
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs = exact_data(case.lhs.space().required_len().unwrap(), 1);
    let rhs = exact_data(case.rhs.space().required_len().unwrap(), 3);
    let resolution = eager_resolution(case);
    let mut digest = FNV_START;
    let mut context = Context::<f64>::default();
    let mut run = |lhs: &[f64], alpha: f64, init: ContractDestinationInit<f64>, out: Vec<f64>| {
        let mut expected = out.clone();
        base_route(
            &resolution,
            dst.space().structure(),
            &mut expected,
            (case.lhs.space().structure(), lhs),
            (case.rhs.space().structure(), &rhs),
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
        assert_eq!(
            context.last_resolution_is_core(),
            route == RouteKind::Core,
            "{route:?}"
        );
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

/// Portable digests (see [`portable_bits`]) of the eager Core / CopyC
/// replays, recorded on `e6e9cac8`.
const EAGER_CORE_PINNED: [(&str, u64); 4] = [
    ("core, inactive block", 0xa9b1_d5db_d80a_fb07),
    ("core, swapped", 0xf11f_58af_16f1_4d74),
    ("copyC C1p", 0x2be7_3679_de6f_8c4e),
    ("copyC C2p", 0x9258_b27c_c589_c856),
];

#[test]
fn eager_core_and_copy_c_bits_are_pinned() {
    let observed = core_fixtures()
        .iter()
        .map(|(name, case, route)| (*name, eager_core_digest(case, *route)))
        .collect::<Vec<_>>();
    assert_eq!(
        observed, EAGER_CORE_PINNED,
        "observed digests: {observed:#x?}"
    );
}

/// The Host member entry over default dense executors, for any route.
/// The route `ContractPlan::new` resolves (the planner with the direct-core
/// executor).
fn member_resolution<R>(case: &Case<R>) -> crate::contract::StorageContractResolution<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    Context::<f64>::default()
        .plan_contract::<crate::DirectCoreExecutor, _>(
            &case.dst(),
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
        )
        .unwrap()
}

fn run_route_members<R>(
    resolution: &crate::contract::StorageContractResolution<f64>,
    case: &Case<R>,
    workspace: &mut crate::contract::HostContractMembersWorkspace<f64>,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: &[f64],
    members: usize,
) -> Result<(), crate::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    run_route_members_with(
        resolution,
        case,
        workspace,
        dst,
        lhs,
        rhs,
        members,
        ContractDestinationInit::Axpby(0.0),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_route_members_with<R>(
    resolution: &crate::contract::StorageContractResolution<f64>,
    case: &Case<R>,
    workspace: &mut crate::contract::HostContractMembersWorkspace<f64>,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: &[f64],
    members: usize,
    init: ContractDestinationInit<f64>,
) -> Result<(), crate::OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    crate::contract::route_host::execute_members_host(
        &mut DenseTreeTransformOperations::default(),
        &mut DenseTreeTransformOperations::default(),
        &mut crate::contract::backend::TensorContractWorkspace::default(),
        &mut FusionBlockContractWorkspace::default(),
        crate::contract::route_host::MemberRoute::Resolution {
            resolution,
            lhs: case.lhs.space().structure(),
            rhs: case.rhs.space().structure(),
        },
        case.dst().space().structure(),
        workspace,
        dst,
        lhs,
        rhs,
        members,
        init,
    )
}

/// `e6e9cac8`'s `ContractPlan` Host member sequence: the exact-sign stacked
/// replay of the core (`zero_inactive`) into the destination, or into a
/// zeroed temporary followed by the member overwrite transform (CopyC).
#[allow(clippy::too_many_arguments)]
fn base_members(
    resolution: &crate::contract::StorageContractResolution<f64>,
    dst_structure: &Arc<symtn_core::BlockStructure>,
    dst: &mut [f64],
    lhs: &[f64],
    rhs: &[f64],
    members: usize,
) {
    use symtn_operations::stacked::{
        StackedDirectReplay, StackedStorageView, StackedStorageViewMut,
    };
    let stacked = |plan: &Arc<symtn_operations::fusion_replay::FusionBlockContractPlan<f64>>,
                   swapped: bool,
                   out: &mut [f64]| {
        let replay = StackedDirectReplay::new_signed(Arc::clone(plan), members).unwrap();
        let [dst_len, left_len, right_len] = replay.member_lens();
        let (left, right) = if swapped { (rhs, lhs) } else { (lhs, rhs) };
        let (left, right) = (left.to_vec(), right.to_vec());
        let mut out_storage = out.to_vec();
        let mut backend = DenseTreeTransformOperations::default();
        let mut backend_workspace = crate::contract::backend::TensorContractWorkspace::default();
        replay
            .execute_signed_host(
                &mut crate::StridedHostKernelAdapter::default(),
                &mut BackendRank2Gemm::<_, _, f64>::new(&mut backend, &mut backend_workspace),
                &mut StackedStorageViewMut::new::<f64>(&mut out_storage, dst_len, members, dst_len)
                    .unwrap(),
                &StackedStorageView::new::<f64>(&left, left_len, members, left_len).unwrap(),
                &StackedStorageView::new::<f64>(&right, right_len, members, right_len).unwrap(),
                true,
            )
            .unwrap();
        out.copy_from_slice(&out_storage);
    };
    match &resolution.route {
        crate::contract::resolution::ContractRoute::Core { plan, swapped } => {
            stacked(plan, *swapped, dst)
        }
        crate::contract::resolution::ContractRoute::CopyC(copy) => {
            let mut temporary = vec![0.0; copy.temporary_len * members];
            stacked(&copy.core, copy.swapped, &mut temporary);
            symtn_operations::tree_transform_members_overwrite_raw(
                &mut crate::StridedHostKernelAdapter::default(),
                &mut symtn_dense::DefaultDenseExecutor::new(),
                &mut symtn_operations::TreeTransformWorkspace::default(),
                &copy.transform,
                dst_structure,
                &copy.temporary,
                dst,
                &temporary,
                members,
                1,
                &[],
            )
            .unwrap();
        }
        crate::contract::resolution::ContractRoute::DynamicTree(_) => {
            panic!("Core / CopyC fixture planned DynamicTree")
        }
    }
}

/// The uniform-twist fZ2×U(1) `mul!` form: its twist rides the Core GEMMs
/// as ±1 per-job alpha.
fn fermionic_core_fixture() -> Case<FermionU1> {
    let fu1 = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let case = Case {
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
    };
    assert_eq!(owned_routes(&case), (RouteKind::Core, RouteKind::Core));
    case
}

fn check_member_bits<R>(name: &str, case: &Case<R>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let resolution = member_resolution(case);
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let mut workspace = crate::contract::HostContractMembersWorkspace::default();
    for members in [1, 2, 17, 1, 2] {
        let lhs = exact_data(lhs_len * members, 1 + members);
        let rhs = exact_data(rhs_len * members, 3 + members);
        let mut actual = vec![f64::NAN; len * members];
        run_route_members(
            &resolution,
            case,
            &mut workspace,
            &mut actual,
            &lhs,
            &rhs,
            members,
        )
        .unwrap();
        let mut expected = vec![f64::NAN; len * members];
        base_members(
            &resolution,
            dst.space().structure(),
            &mut expected,
            &lhs,
            &rhs,
            members,
        );
        assert_eq!(bits(&actual), bits(&expected), "{name}, B = {members}");
    }
}

#[test]
fn member_core_and_copy_c_replays_match_the_base_member_sequence() {
    // What: on one workspace across B = 1, 2, 17, 1, 2, every Core and
    // CopyC member replay is bit-identical, on this machine, to the
    // `ContractPlan` Host sequence it replaces (exact data; NaN-poisoned
    // destinations; a retained CopyC temporary across B changes).
    for (name, case, _) in core_fixtures() {
        check_member_bits(name, &case);
    }
    check_member_bits("fermionic core", &fermionic_core_fixture());
}

fn check_member_oracle<R>(name: &str, case: &Case<R>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn_core::PhysicalFusionBasis<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let resolution = member_resolution(case);
    assert!(!resolution.is_dynamic_tree(), "{name}");
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let mut workspace = crate::contract::HostContractMembersWorkspace::default();
    for members in [1, 2, 17] {
        let lhs = (0..members)
            .flat_map(|i| host_data(case.lhs.space(), 3 + 2 * i))
            .collect::<Vec<_>>();
        let rhs = (0..members)
            .flat_map(|i| host_data(case.rhs.space(), 7 + 4 * i))
            .collect::<Vec<_>>();
        let mut actual = vec![f64::NAN; len * members];
        run_route_members(
            &resolution,
            case,
            &mut workspace,
            &mut actual,
            &lhs,
            &rhs,
            members,
        )
        .unwrap();
        for member in [0, members / 2, members - 1] {
            let (oracle_shape, oracle) = physical_oracle(
                case,
                &lhs[member * lhs_len..(member + 1) * lhs_len],
                &rhs[member * rhs_len..(member + 1) * rhs_len],
            );
            let (shape, found) = crate::expand_physical_host(
                crate::BoundDynamicTensorRef::try_new(
                    &dst,
                    &actual[member * len..(member + 1) * len],
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(shape, oracle_shape, "{name}");
            for (&value, &want) in found.iter().zip(&oracle) {
                assert!(
                    (value - want).abs() < 1e-9 * (1.0 + want.abs()),
                    "{name}, B = {members}, member {member}: {value} vs {want}"
                );
            }
        }
    }
}

#[test]
fn member_core_and_copy_c_replays_match_the_dense_oracle() {
    // What: each member of a B = 1, 2, 17 stack equals the dense
    // physical-basis contraction of its own operands (an oracle independent
    // of every replay path), U(1) and SU(2), Core and CopyC.
    for (name, case, _) in core_fixtures() {
        check_member_oracle(name, &case);
    }
    let su2 = Arc::new(SU2FusionRule);
    let su2_copy_c = square_case(&su2, su2_leg, [3, 2], [1, 0], [1, 0, 3, 2]);
    assert_eq!(
        owned_routes(&su2_copy_c),
        (RouteKind::CopyC, RouteKind::CopyC)
    );
    check_member_oracle("SU2 copyC C1p", &su2_copy_c);
    let su2_core = square_case(&su2, su2_leg, [2, 3], [0, 1], [0, 1, 2, 3]);
    assert_eq!(owned_routes(&su2_core), (RouteKind::Core, RouteKind::Core));
    check_member_oracle("SU2 core", &su2_core);
}

#[test]
fn a_failed_copy_c_member_replay_refills_its_temporary_on_retry() {
    // Why: CopyC's temporary shares the workspace core destination and its
    // #1747 proof. A replay that dirties it and fails (the fault hook) must
    // leave the retry to refill the inactive blocks; a warm replay of the
    // same plan does not refill.
    let (_, case, route) = core_fixtures().remove(2);
    assert_eq!(route, RouteKind::CopyC);
    let resolution = member_resolution(&case);
    let dst = case.dst();
    let len = dst.space().required_len().unwrap();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let mut workspace = crate::contract::HostContractMembersWorkspace::default();
    let fills = |members: usize, poison: bool, workspace: &mut _| {
        let lhs = exact_data(lhs_len * members, 1);
        let rhs = exact_data(rhs_len * members, 3);
        let mut actual = vec![f64::NAN; len * members];
        let mut expected = vec![f64::NAN; len * members];
        base_members(
            &resolution,
            dst.space().structure(),
            &mut expected,
            &lhs,
            &rhs,
            members,
        );
        let workspace: &mut crate::contract::HostContractMembersWorkspace<f64> = workspace;
        if poison {
            workspace.poison_dst_then_fail();
        }
        let before = workspace.core_inactive_fills();
        let result = run_route_members(
            &resolution,
            &case,
            workspace,
            &mut actual,
            &lhs,
            &rhs,
            members,
        );
        if poison {
            assert!(result.is_err());
            assert!(actual.iter().all(|value| value.is_nan()));
            return None;
        }
        result.unwrap();
        assert_eq!(bits(&actual), bits(&expected), "B = {members}");
        Some(workspace.core_inactive_fills() - before)
    };
    for members in [2, 1] {
        assert_eq!(fills(members, false, &mut workspace), Some(1));
        assert_eq!(fills(members, false, &mut workspace), Some(0));
        let mut fresh = crate::contract::HostContractMembersWorkspace::default();
        assert_eq!(fills(members, false, &mut fresh), Some(1));
        // A fresh workspace dirtied by the fault hook before its first
        // proof refills on retry.
        let mut dirty = crate::contract::HostContractMembersWorkspace::default();
        assert_eq!(fills(members, true, &mut dirty), None);
        assert_eq!(fills(members, false, &mut dirty), Some(1));
        assert_eq!(fills(members, false, &mut dirty), Some(0));
    }
}

/// Bits with `-0` folded into `+0`: the overwrite kernel accumulates onto a
/// zero pass, so it may turn a `-0` the `beta = 0` kernel writes into `+0`.
fn folded_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| (value + 0.0).to_bits()).collect()
}

#[test]
fn zeroed_member_replays_skip_the_fills_of_a_born_zero_destination() {
    // What (#2123): into a destination born zero and reused across
    // B = 1, 2, 17, 1 with changed inputs on every call, `Zeroed` (the
    // `ContractPlan::execute` init) gives the bits `Axpby(0)` gives into a
    // NaN-poisoned destination. A directly written destination gets no
    // inactive fill under `Zeroed` and one per call under `Axpby(0)`; a
    // workspace core destination's fills do not depend on `init`; the
    // `B = 1` output transform runs the overwrite kernel exactly under
    // `Zeroed`.
    let mut fixtures: Vec<_> = core_fixtures()
        .into_iter()
        .map(|(name, case, _)| (name, case))
        .collect();
    fixtures.push((
        "dynamicTree identity output, inactive block",
        overwrite_cases().remove(1).1,
    ));
    fixtures.push(("dynamicTree source and output transforms", u1_case()));
    for (name, case) in fixtures {
        let resolution = member_resolution(&case);
        let route = crate::contract::route_host::RouteView::of(
            &resolution.route,
            case.lhs.space().structure(),
            case.rhs.space().structure(),
        );
        let direct = route.output.is_none();
        if name.contains("inactive") {
            assert!(
                !route.plan.inactive_destination_regions().is_empty(),
                "{name}"
            );
        }
        assert_eq!(
            resolution.is_dynamic_tree(),
            name.starts_with("dynamicTree"),
            "{name}"
        );
        let len = case.dst().space().required_len().unwrap();
        let lhs_len = case.lhs.space().required_len().unwrap();
        let rhs_len = case.rhs.space().required_len().unwrap();
        let mut zeroed = crate::contract::HostContractMembersWorkspace::default();
        let mut axpby = crate::contract::HostContractMembersWorkspace::default();
        let counters = |workspace: &crate::contract::HostContractMembersWorkspace<f64>| {
            [
                workspace.core_inactive_fills(),
                workspace.output_overwrites(),
            ]
        };
        let mut born = Vec::new();
        for (call, members) in [1, 2, 17, 1].into_iter().enumerate() {
            if born.len() != len * members {
                born = vec![0.0; len * members];
            }
            for salt in [0, 1] {
                let lhs = exact_data(lhs_len * members, 1 + 2 * call + salt);
                let rhs = exact_data(rhs_len * members, 2 + call + 3 * salt);
                let before = [counters(&zeroed), counters(&axpby)];
                run_route_members_with(
                    &resolution,
                    &case,
                    &mut zeroed,
                    &mut born,
                    &lhs,
                    &rhs,
                    members,
                    ContractDestinationInit::Zeroed,
                )
                .unwrap();
                let mut expected = vec![f64::NAN; len * members];
                run_route_members_with(
                    &resolution,
                    &case,
                    &mut axpby,
                    &mut expected,
                    &lhs,
                    &rhs,
                    members,
                    ContractDestinationInit::Axpby(0.0),
                )
                .unwrap();
                let what = format!("{name}, B = {members}, call {salt}");
                assert_eq!(folded_bits(&born), folded_bits(&expected), "{what}");
                let [z, a] = [
                    (counters(&zeroed), before[0]),
                    (counters(&axpby), before[1]),
                ]
                .map(|(after, before)| [after[0] - before[0], after[1] - before[1]]);
                if direct {
                    assert_eq!((z, a), ([0, 0], [1, 0]), "{what}");
                } else {
                    assert_eq!(z[0], a[0], "{what}");
                    assert_eq!((z[1], a[1]), (usize::from(members == 1), 0), "{what}");
                }
            }
        }
    }
}
