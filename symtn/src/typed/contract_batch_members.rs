//! #1859 Host H2: `ContractPlan` Host member replays of the Core and CopyC
//! routes on one plan and workspace across B = 1, 2, 17, 1, into fresh and
//! NaN-poisoned destinations, against the eager `contract` of each member's
//! own operands (independently checked against TensorKit's `blas_contract!`
//! steps and the physical-basis contraction in
//! `tests/typed_contract_host_oracle.rs`).

use super::*;
use crate::sector::{U1FusionRule, U1Irrep};
use crate::typed::{ContractSpec, GradedSpace, TensorMap};

/// Exactly representable payload in `{-2, -1, ±0, 1, 2}`.
fn exact(len: usize, salt: usize) -> Vec<f64> {
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

/// Every member of `dst` equals the eager contraction of that member's
/// operands; exact data, so up to the sign of an exact zero.
fn assert_members_are_eager(
    engine: &mut dyn Engine,
    what: &str,
    dst: &StackedTensorMap<U1FusionRule, f64>,
    lhs: &StackedTensorMap<U1FusionRule, f64>,
    rhs: &StackedTensorMap<U1FusionRule, f64>,
    spec: &ContractSpec<'_>,
    members: usize,
) {
    for member in 0..members {
        let rhs_member = &rhs.member(engine, member).unwrap();
        let eager = lhs
            .member(engine, member)
            .unwrap()
            .contract(engine, rhs_member, spec)
            .unwrap();
        let expected = eager.dense_data().unwrap();
        assert!(expected.iter().any(|value| *value != 0.0), "{what}");
        let actual = dst.member(engine, member).unwrap();
        assert_eq!(
            actual.dense_data().unwrap(),
            expected,
            "{what}, B = {members}, member {member}"
        );
    }
}

fn check_members(
    engine: &mut dyn Engine,
    name: &str,
    lhs_axes: &[usize],
    codomain: &[usize],
    copy_c: bool,
) {
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap();
    let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v, &v], [&w], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&w], [&v], 6).unwrap();
    let spec = ContractSpec {
        lhs: lhs_axes,
        rhs: &[0],
        codomain,
        domain: &[2],
    };
    let mut plan_and_workspace = None;
    for members in [1, 2, 17, 1] {
        let mut lhs = StackedTensorMap::pack(&vec![&a; members]).unwrap();
        let mut rhs = StackedTensorMap::pack(&vec![&b; members]).unwrap();
        let lhs_len = lhs.storage.len();
        let rhs_len = rhs.storage.len();
        lhs.storage = exact(lhs_len, 1);
        rhs.storage = exact(rhs_len, 3);
        let (plan, workspace) = plan_and_workspace.get_or_insert_with(|| {
            let plan = ContractPlan::new(engine, &lhs, &rhs, &spec).unwrap();
            let workspace = plan.workspace().unwrap();
            (plan, workspace)
        });
        assert_eq!(plan.copy_c().is_some(), copy_c, "{name}");
        assert!(!plan.resolution.is_dynamic_tree(), "{name}");
        plan.execute(engine, &lhs, &rhs, workspace).unwrap();
        let mut dst = workspace.take_output().unwrap();
        assert_members_are_eager(engine, name, &dst, &lhs, &rhs, &spec, members);
        dst.storage.fill(f64::NAN);
        plan.execute_into(engine, &lhs, &rhs, &mut dst, workspace)
            .unwrap();
        assert_members_are_eager(engine, name, &dst, &lhs, &rhs, &spec, members);
    }
}

#[test]
fn member_core_and_copy_c_replays_equal_the_eager_contraction_per_member() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    check_members(engine, "core", &[2], &[0, 1], false);
    check_members(engine, "copyC", &[2], &[1, 0], true);
}
