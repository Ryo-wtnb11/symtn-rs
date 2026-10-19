mod common;
#[path = "../../tests/support"]
mod support {
    use num_complex::{Complex32, Complex64};
    pub mod numerics;
}
#[path = "support/report.rs"]
mod report;
use report::Report;

use support::numerics;
#[macro_use]
#[allow(unused_macros)]
mod contract_cases;
mod braiding_probe;

use braiding_probe::{ProbeSector, RealBraidingProbe};
use contract_cases::{
    blas_contract_oracle, candidate_core_probes, dense_oracle, fermion_su2, fermion_u1,
    fermionic_blas_contract_oracle, fermionic_blas_contract_oracle_partitioned,
    fermionic_canonical_nonuniform, fermionic_twist_roles, fill, poisoned_destination, su2,
    su2_bent, su2_reordered, u1, u1_inactive_cases, u1_non_self_dual, u1_reordered,
    u1_rhs_identity, Case, Payload, TwistRole,
};
use std::sync::Arc;
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, PhysicalFusionBasis, ProductFusionRuleExt,
    SectorCodec,
};
use symtn::typed::GradedSpace;
use symtn::typed::{ContractPlan, ContractSpec, Error, Runtime, StackedTensorMap, TensorMap};

fn check<R, D>(case: Case<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| {
                let scale = D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0);
                let lhs = case.lhs.scale(scale).unwrap();
                let rhs = case
                    .rhs
                    .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                    .unwrap();
                Case {
                    name: case.name,
                    lhs,
                    rhs,
                    lhs_axes: case.lhs_axes.clone(),
                    rhs_axes: case.rhs_axes.clone(),
                    output_axes: case.output_axes.clone(),
                    dense: case.dense,
                }
            })
            .collect();
        let left: Vec<_> = members.iter().map(|member| &member.lhs).collect();
        let right: Vec<_> = members.iter().map(|member| &member.rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap();
        let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let result = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        for (i, member_case) in members.iter().enumerate() {
            let member = result.member(i).unwrap();
            let reference = blas_contract_oracle(member_case);
            numerics::assert_nonzero_slices_close(
                case.name,
                member.dense_data().unwrap(),
                reference.dense_data().unwrap(),
                case.terms(),
            );
            if member_case.dense {
                let (_, expected) = dense_oracle(member_case);
                let physical = member.to_physical_dense().unwrap();
                numerics::assert_nonzero_slices_close(
                    case.name,
                    &physical.data,
                    &expected,
                    case.terms(),
                );
            }
            let eager = member_case.host();
            assert_eq!(member.codomain_rank(), eager.codomain_rank());
            assert_eq!(
                result.signature(),
                StackedTensorMap::pack(&[&eager]).unwrap().signature(),
                "{} member {i}: full output structure differs",
                case.name,
            );
            numerics::assert_nonzero_slices_close(
                case.name,
                member.dense_data().unwrap(),
                eager.dense_data().unwrap(),
                case.terms(),
            );
        }
        let poison: Vec<_> = members.iter().map(poisoned_destination).collect();
        let mut destination = StackedTensorMap::pack(&poison).unwrap();
        plan.execute_into(&lhs, &rhs, &mut destination, &mut workspace)
            .unwrap();
        for (i, member_case) in members.iter().enumerate() {
            let member = destination.member(i).unwrap();
            let reference = blas_contract_oracle(member_case);
            numerics::assert_nonzero_slices_close(
                case.name,
                member.dense_data().unwrap(),
                reference.dense_data().unwrap(),
                case.terms(),
            );
            if member_case.dense {
                let (_, expected) = dense_oracle(member_case);
                numerics::assert_nonzero_slices_close(
                    case.name,
                    &member.to_physical_dense().unwrap().data,
                    &expected,
                    case.terms(),
                );
            }
        }
    }
}

#[test]
fn public_host_batch_matches_reference_steps_and_applicable_physical_basis() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    check(u1_reordered::<f64>(&runtime));
    check(su2_reordered::<f64>(&runtime));
    check(u1_reordered::<symtn::typed::Complex64>(&runtime));
    check(su2_reordered::<symtn::typed::Complex64>(&runtime));
    check(u1_reordered::<f32>(&runtime));
    check(su2_reordered::<symtn::typed::Complex32>(&runtime));
    check(
        u1_inactive_cases::<f64>(&runtime)
            .into_iter()
            .nth(1)
            .unwrap(),
    );
    check(su2_bent::<f64>(&runtime));
    check(u1_rhs_identity::<f64>(&runtime));
}

#[test]
fn public_host_batch_core_and_swapped_core_match_reference_steps() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    fn probes<R, D>(runtime: &Runtime, v: &GradedSpace<R>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + PhysicalFusionBasis<Scalar = f64>,
        D: Payload,
    {
        for (mut case, _) in candidate_core_probes::<R, D>(runtime, v)
            .into_iter()
            .filter(|(case, _)| matches!(case.name, "C0" | "C2"))
        {
            case.dense = case.name == "C0";
            check(case);
        }
    }
    probes::<_, f64>(&runtime, &u1_non_self_dual());
    probes::<_, symtn::typed::Complex64>(&runtime, &u1_non_self_dual());
    probes::<_, f64>(&runtime, &su2());
    probes::<_, symtn::typed::Complex64>(&runtime, &su2());
    check(
        u1_inactive_cases::<f64>(&runtime)
            .into_iter()
            .next()
            .unwrap(),
    );
}

#[test]
fn swapped_core_uses_rhs_member_length_and_nondefault_output_split() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = u1(&[(0, 2), (1, 1)]);
    let w = u1(&[(0, 1), (1, 2)]);
    let x = u1(&[(0, 2), (1, 3)]);
    let lhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&w], fill(171)).unwrap();
    let rhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&x, &x], [&v], fill(172)).unwrap();
    let spec = ContractSpec {
        lhs: &[0],
        rhs: &[2],
        codomain: &[1, 2],
        domain: &[0],
    };
    let eager = lhs.contract(&rhs, &spec).unwrap();
    let reference = rhs.compose(&lhs).unwrap();
    numerics::assert_nonzero_slices_close(
        "swapped eager",
        eager.dense_data().unwrap(),
        reference.dense_data().unwrap(),
        64,
    );
    let left = StackedTensorMap::pack(&[&lhs, &lhs]).unwrap();
    let right = StackedTensorMap::pack(&[&rhs, &rhs]).unwrap();
    assert_ne!(
        lhs.dense_data().unwrap().len(),
        rhs.dense_data().unwrap().len()
    );
    let plan = ContractPlan::new(&left, &right, &spec).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let actual = plan.execute(&left, &right, &mut workspace).unwrap();
    for i in 0..2 {
        assert_eq!(actual.member(i).unwrap().codomain_rank(), 2);
        numerics::assert_nonzero_slices_close(
            "swapped batch",
            actual.member(i).unwrap().dense_data().unwrap(),
            reference.dense_data().unwrap(),
            64,
        );
    }
}

#[test]
fn binding_errors_do_not_write_destination_and_workspaces_are_independent() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    binding_errors_case(&runtime, u1_reordered::<f64>(&runtime));
    let core = candidate_core_probes::<_, f64>(&runtime, &u1_non_self_dual())
        .into_iter()
        .find(|(case, _)| case.name == "C2")
        .unwrap()
        .0;
    binding_errors_case(&runtime, core);
    let v = u1(&[(0, 2), (1, 2)]);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 3).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 4).unwrap();
    binding_errors_case(
        &runtime,
        Case {
            name: "copyC bindings",
            lhs: a,
            rhs: b,
            lhs_axes: vec![2, 3],
            rhs_axes: vec![0, 1],
            output_axes: vec![1, 0, 2, 3],
            dense: false,
        },
    );
}

fn binding_errors_case(runtime: &Runtime, case: Case<symtn::sector::U1FusionRule, f64>) {
    let lhs = StackedTensorMap::pack(&[&case.lhs, &case.lhs]).unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs, &case.rhs]).unwrap();
    let plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let other_plan = ContractPlan::new(&lhs, &rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut second = plan.workspace().unwrap();
    let mut foreign = other_plan.workspace().unwrap();
    let mut destination =
        StackedTensorMap::pack(&[poisoned_destination(&case), poisoned_destination(&case)])
            .unwrap();
    let bits = |stack: &StackedTensorMap<_, f64>| {
        (0..stack.len())
            .flat_map(|i| {
                stack
                    .member(i)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let before = bits(&destination);
    let wrong_count = lhs.select(&[0]).unwrap();
    assert!(
        matches!(ContractPlan::new(&wrong_count, &rhs, &case.spec()), Err(Error::InvalidArgument(m)) if m.contains("operand stacks have different member counts"))
    );
    assert!(plan
        .execute_into(&wrong_count, &rhs, &mut destination, &mut workspace)
        .err()
        .unwrap()
        .report()
        .contains("operand stacks have different member counts"));
    assert_eq!(bits(&destination), before);
    assert!(plan
        .execute_into(&lhs, &rhs, &mut destination, &mut foreign)
        .err()
        .unwrap()
        .report()
        .contains("workspace belongs to another plan"));
    assert_eq!(bits(&destination), before);
    let wrong_destination = destination.select(&[0]).unwrap();
    let mut wrong_destination = wrong_destination;
    let wrong_before = bits(&wrong_destination);
    assert!(plan
        .execute_into(&lhs, &rhs, &mut wrong_destination, &mut workspace)
        .err()
        .unwrap()
        .report()
        .contains("destination member count differs"));
    assert_eq!(bits(&wrong_destination), wrong_before);
    let v = u1(&[(0, 1)]);
    let wrong_member = TensorMap::<_, f64>::zeros(runtime, [&v], [&v]).unwrap();
    let wrong = StackedTensorMap::pack(&[wrong_member.clone(), wrong_member]).unwrap();
    assert!(matches!(
        plan.execute_into(&wrong, &rhs, &mut destination, &mut workspace),
        Err(symtn::typed::Error::BatchSignatureMismatch { member: None, .. })
    ));
    assert_eq!(bits(&destination), before);
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();
    plan.execute(&lhs, &rhs, &mut second).unwrap();
    assert!(workspace.retained_bytes() > 0);
    assert!(workspace.take_output().is_some());
    assert!(workspace.take_output().is_none());
    plan.execute_into(&lhs, &rhs, &mut destination, &mut second)
        .unwrap();
    let singleton = rhs.select(&[0]).unwrap();
    let singleton_lhs = lhs.select(&[0]).unwrap();
    plan.execute(&singleton_lhs, &singleton, &mut workspace)
        .unwrap();
    let many = vec![0; 17];
    let many_lhs = lhs.select(&many).unwrap();
    let many_rhs = rhs.select(&many).unwrap();
    assert_eq!(
        plan.execute(&many_lhs, &many_rhs, &mut workspace)
            .unwrap()
            .len(),
        17
    );
    assert_eq!(plan.execute(&lhs, &rhs, &mut second).unwrap().len(), 2);
}

#[test]
fn public_copy_c_matches_eager() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = u1(&[(0, 2), (1, 2)]);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 3).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v, &v], 4).unwrap();
    let lhs = StackedTensorMap::pack(&[&a]).unwrap();
    let rhs = StackedTensorMap::pack(&[&b]).unwrap();
    for split in [1, 2, 3] {
        let output = [1, 0, 2, 3];
        let (codomain, domain) = output.split_at(split);
        let spec = ContractSpec {
            lhs: &[2, 3],
            rhs: &[0, 1],
            codomain,
            domain,
        };
        let expected = a.contract(&b, &spec).unwrap();
        let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
        let result = plan
            .execute(&lhs, &rhs, &mut plan.workspace().unwrap())
            .unwrap()
            .member(0)
            .unwrap();
        assert_eq!(result.codomain_rank(), split);
        assert_eq!(
            StackedTensorMap::pack(&[&expected]).unwrap().signature(),
            StackedTensorMap::pack(&[&result]).unwrap().signature(),
        );
        numerics::assert_nonzero_slices_close(
            "copyC",
            result.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            16,
        );
    }
}

#[test]
fn public_copy_c_both_orientations_and_dtypes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    fn cases<R, D>(runtime: &Runtime, v: &GradedSpace<R>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + PhysicalFusionBasis<Scalar = f64>,
        D: Payload,
    {
        for (mut case, _) in candidate_core_probes::<R, D>(runtime, v) {
            match case.name {
                "C1" => {
                    case.name = "C1p";
                    case.output_axes = vec![1, 0, 3, 2];
                }
                "C2" => {
                    case.name = "C2p";
                    case.output_axes = vec![3, 2, 1, 0];
                }
                _ => continue,
            }
            check(case);
        }
    }
    cases::<_, f64>(&runtime, &u1_non_self_dual());
    cases::<_, symtn::typed::Complex64>(&runtime, &u1_non_self_dual());
    cases::<_, f32>(&runtime, &u1_non_self_dual());
    cases::<_, symtn::typed::Complex32>(&runtime, &u1_non_self_dual());
    cases::<_, f64>(&runtime, &su2());
    cases::<_, symtn::typed::Complex64>(&runtime, &su2());
    cases::<_, f32>(&runtime, &su2());
    cases::<_, symtn::typed::Complex32>(&runtime, &su2());
}

#[test]
fn public_copy_c_small_output_winners() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    fn cases<R, D>(runtime: &Runtime, v: &GradedSpace<R>, c: &GradedSpace<R>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec
            + PhysicalFusionBasis<Scalar = f64>,
        D: Payload,
    {
        let tensor = |codomain: [&GradedSpace<R>; 2], domain: [&GradedSpace<R>; 2], salt| {
            TensorMap::<R, D>::from_subblock_fn(runtime, codomain, domain, fill(salt)).unwrap()
        };
        check(Case {
            name: "S1",
            lhs: tensor([c, c], [v, v], 101),
            rhs: tensor([v, v], [c, c], 102),
            lhs_axes: vec![3, 2],
            rhs_axes: vec![1, 0],
            output_axes: vec![2, 3, 0, 1],
            dense: false,
        });
        check(Case {
            name: "S2",
            lhs: tensor([v, v], [c, c], 103),
            rhs: tensor([c, c], [v, v], 104),
            lhs_axes: vec![0, 1],
            rhs_axes: vec![2, 3],
            output_axes: vec![3, 2, 1, 0],
            dense: false,
        });
    }
    let v = u1(&[(-1, 8), (0, 8), (1, 8)]);
    let c = u1(&[(0, 2), (1, 1)]);
    cases::<_, f64>(&runtime, &v, &c);
    cases::<_, symtn::typed::Complex64>(&runtime, &v, &c);
    let c_su2 = GradedSpace::try_new(
        Arc::new(symtn::sector::SU2FusionRule),
        [(symtn::sector::SU2Irrep::from_twice_spin(0), 1)],
    )
    .unwrap();
    cases::<_, f64>(&runtime, &su2(), &c_su2);
    cases::<_, symtn::typed::Complex64>(&runtime, &su2(), &c_su2);
}

#[test]
fn non_symmetric_braiding_is_rejected_before_plan_compile() {
    let runtime = Runtime::builder().build().unwrap();
    for anyonic in [false, true] {
        let rejected = if anyonic {
            reject_braiding::<true>(&runtime)
        } else {
            reject_braiding::<false>(&runtime)
        };
        assert!(rejected);
    }
}

#[test]
fn twist_bearing_dynamic_tree_matches_literal_tensorkit_steps() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! cases {
        ($rule:ty, $dtype:ty, $space:expr) => {{
            let space = $space;
            let twist = |tensor: &TensorMap<$rule, $dtype>, legs: &[usize]| {
                tensor
                    .twist(legs, symtn::typed::Direction::Forward)
                    .unwrap()
            };
            for (case, role) in fermionic_twist_roles::<$rule, $dtype>(
                &runtime,
                &space,
                ["A", "canonical", "B", "both"],
                71,
            )
            .into_iter()
            .zip([TwistRole::A, TwistRole::A, TwistRole::B, TwistRole::A])
            {
                check_twisted_members(case, role, twist);
            }
            check_twisted_members(
                fermionic_canonical_nonuniform::<$rule, $dtype>(
                    &runtime,
                    &space,
                    "canonical nonuniform",
                    39,
                ),
                TwistRole::A,
                twist,
            );
        }};
    }
    cases!(contract_cases::FermionU1, f64, fermion_u1());
    cases!(
        contract_cases::FermionU1,
        symtn::typed::Complex64,
        fermion_u1()
    );
    cases!(contract_cases::FermionSu2, f64, fermion_su2());
    cases!(
        contract_cases::FermionSu2,
        symtn::typed::Complex64,
        fermion_su2()
    );
}

fn check_twisted_members<R, D>(
    case: Case<R, D>,
    role: TwistRole,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    fn compare<D: Payload>(actual: &[D], expected: &[D], terms: usize, what: &str) {
        assert_eq!(actual.len(), expected.len(), "{what}: payload length");
        let scale = expected
            .iter()
            .map(|value| value.magnitude())
            .fold(1.0, f64::max);
        let tolerance = 32.0 * (terms.max(1) as f64).sqrt() * D::EPS * scale;
        for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.distance(b) <= tolerance,
                "{what} [{}] element {index}: {a:?} versus {b:?}, tolerance {tolerance:e}",
                D::NAME
            );
        }
    }
    let first_lhs = StackedTensorMap::pack(&[&case.lhs]).unwrap();
    let first_rhs = StackedTensorMap::pack(&[&case.rhs]).unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &case.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut independent = plan.workspace().unwrap();
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| Case {
                name: case.name,
                lhs: case
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0))
                    .unwrap(),
                rhs: case
                    .rhs
                    .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                    .unwrap(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: false,
            })
            .collect();
        let lhs_refs: Vec<_> = members.iter().map(|member| &member.lhs).collect();
        let rhs_refs: Vec<_> = members.iter().map(|member| &member.rhs).collect();
        let lhs = StackedTensorMap::pack(&lhs_refs).unwrap();
        let rhs = StackedTensorMap::pack(&rhs_refs).unwrap();
        let output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        for (i, member) in members.iter().enumerate() {
            let actual = output.member(i).unwrap();
            let eager = member.host();
            let oracle = fermionic_blas_contract_oracle(member, role, twist);
            // Every output entry is bilinear, so the product of reduced source
            // lengths bounds all contracted and recoupled floating terms.
            let terms =
                member.lhs.dense_data().unwrap().len() * member.rhs.dense_data().unwrap().len();
            assert_eq!(
                output.signature(),
                StackedTensorMap::pack(&[&eager]).unwrap().signature(),
                "{}: output structure",
                case.name
            );
            compare(
                actual.dense_data().unwrap(),
                eager.dense_data().unwrap(),
                terms,
                case.name,
            );
            compare(
                actual.dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                terms,
                case.name,
            );
            if i == 0 {
                let untwisted = fermionic_blas_contract_oracle(member, TwistRole::None, twist);
                let expected = oracle.dense_data().unwrap();
                let scale = expected
                    .iter()
                    .map(|value| value.magnitude())
                    .fold(0.0, f64::max);
                let tolerance = 32.0 * (terms.max(1) as f64).sqrt() * D::EPS * scale.max(1.0);
                assert!(
                    expected
                        .iter()
                        .zip(untwisted.dense_data().unwrap())
                        .any(|(&a, &b)| a.distance(b) > tolerance),
                    "{}: twist is vacuous",
                    case.name
                );
            }
        }
        let poisoned: Vec<_> = members.iter().map(poisoned_destination).collect();
        let refs: Vec<_> = poisoned.iter().collect();
        let mut dst = StackedTensorMap::pack(&refs).unwrap();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut independent)
            .unwrap();
        for (i, member) in members.iter().enumerate() {
            compare(
                dst.member(i).unwrap().dense_data().unwrap(),
                member.host().dense_data().unwrap(),
                member.lhs.dense_data().unwrap().len() * member.rhs.dense_data().unwrap().len(),
                case.name,
            );
        }
        if count == 2 {
            let before: Vec<_> = (0..count)
                .map(|i| dst.member(i).unwrap().dense_data().unwrap().to_vec())
                .collect();
            let short_rhs = StackedTensorMap::pack(&[&members[0].rhs]).unwrap();
            assert!(plan
                .execute_into(&lhs, &short_rhs, &mut dst, &mut independent)
                .is_err());
            for (i, data) in before.iter().enumerate() {
                assert_eq!(dst.member(i).unwrap().dense_data().unwrap(), data);
            }
        }
    }
}

fn check_fermionic_unit_batch<R, D>(
    case: Case<R, D>,
    split: usize,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| Case {
                name: case.name,
                lhs: case
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0))
                    .unwrap(),
                rhs: case
                    .rhs
                    .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                    .unwrap(),
                lhs_axes: case.lhs_axes.clone(),
                rhs_axes: case.rhs_axes.clone(),
                output_axes: case.output_axes.clone(),
                dense: false,
            })
            .collect();
        let left: Vec<_> = members.iter().map(|member| &member.lhs).collect();
        let right: Vec<_> = members.iter().map(|member| &member.rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap();
        let spec = ContractSpec {
            lhs: &case.lhs_axes,
            rhs: &case.rhs_axes,
            codomain: &case.output_axes[..split],
            domain: &case.output_axes[split..],
        };
        let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let result = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        for (index, member) in members.iter().enumerate() {
            let actual = result.member(index).unwrap();
            // The selected swapped core absorbs the orientation into its
            // structural lowering. TensorKit's unswapped step sequence twists A.
            let role = if case.name == "C2" {
                TwistRole::A
            } else {
                TwistRole::None
            };
            let oracle = fermionic_blas_contract_oracle_partitioned(member, role, split, twist);
            if case.name == "C2" && count == 1 {
                let untwisted = fermionic_blas_contract_oracle_partitioned(
                    member,
                    TwistRole::None,
                    split,
                    twist,
                );
                let scale = oracle
                    .dense_data()
                    .unwrap()
                    .iter()
                    .map(|value| value.magnitude())
                    .fold(0.0_f64, f64::max);
                assert!(oracle
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(untwisted.dense_data().unwrap())
                    .any(|(&a, &b)| a.distance(b) > 1e-3 * scale));
            }
            let eager = member.lhs.contract(&member.rhs, &spec).unwrap();
            assert_eq!(actual.codomain_rank(), split);
            assert_eq!(
                result.signature(),
                StackedTensorMap::pack(&[&eager]).unwrap().signature(),
                "{} member {index}: output structure",
                case.name
            );
            numerics::assert_nonzero_slices_close(
                case.name,
                actual.dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                case.terms(),
            );
            numerics::assert_nonzero_slices_close(
                case.name,
                actual.dense_data().unwrap(),
                eager.dense_data().unwrap(),
                case.terms(),
            );
        }
        if count == 2 {
            let outputs: Vec<_> = members
                .iter()
                .map(|member| member.lhs.contract(&member.rhs, &spec).unwrap())
                .collect();
            let refs: Vec<_> = outputs.iter().collect();
            let mut destination = StackedTensorMap::pack(&refs).unwrap();
            let before: Vec<Vec<D>> = (0..count)
                .map(|i| {
                    destination
                        .member(i)
                        .unwrap()
                        .dense_data()
                        .unwrap()
                        .to_vec()
                })
                .collect();
            let short_rhs = StackedTensorMap::pack(&[&members[0].rhs]).unwrap();
            assert!(plan
                .execute_into(&lhs, &short_rhs, &mut destination, &mut workspace)
                .is_err());
            for (i, expected) in before.iter().enumerate() {
                assert_eq!(
                    destination.member(i).unwrap().dense_data().unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
fn fermionic_unit_direct_and_copy_c_match_tensorkit_steps() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! cases {
        ($rule:ty, $dtype:ty, $space:expr) => {{
            let space = $space;
            let twist = |t: &TensorMap<$rule, $dtype>, legs: &[usize]| {
                t.twist(legs, symtn::typed::Direction::Forward).unwrap()
            };
            for (mut case, _) in candidate_core_probes::<$rule, $dtype>(&runtime, &space) {
                if !matches!(case.name, "C0" | "C1" | "C2") {
                    continue;
                }
                // The output transform owns fermionic permutation signs.
                // A different codomain split also exercises destination space construction.
                if matches!(case.name, "C1" | "C2") {
                    let plain = Case {
                        name: case.name,
                        lhs: case.lhs.clone(),
                        rhs: case.rhs.clone(),
                        lhs_axes: case.lhs_axes.clone(),
                        rhs_axes: case.rhs_axes.clone(),
                        output_axes: case.output_axes.clone(),
                        dense: false,
                    };
                    check_fermionic_unit_batch(plain, 2, twist);
                    case.output_axes = if case.name == "C1" {
                        vec![1, 0, 3, 2]
                    } else {
                        vec![3, 2, 1, 0]
                    };
                    check_fermionic_unit_batch(case, 1, twist);
                } else {
                    check_fermionic_unit_batch(case, 2, twist);
                }
            }
        }};
    }
    cases!(contract_cases::FermionU1, f64, fermion_u1());
    cases!(
        contract_cases::FermionU1,
        symtn::typed::Complex64,
        fermion_u1()
    );
    cases!(contract_cases::FermionSu2, f64, fermion_su2());
    cases!(
        contract_cases::FermionSu2,
        symtn::typed::Complex64,
        fermion_su2()
    );
}

fn check_mixed_signed_members<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    swapped: bool,
    twist: impl Fn(&TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let dual = v.try_dual().unwrap();
    let lhs = TensorMap::<R, D>::from_subblock_fn(runtime, [v], [&dual], fill(201)).unwrap();
    let rhs = TensorMap::<R, D>::from_subblock_fn(runtime, [&dual], [v], fill(202)).unwrap();
    let base = Case {
        name: if swapped {
            "mixed signed swapped"
        } else {
            "mixed signed direct"
        },
        lhs: if swapped { rhs.clone() } else { lhs.clone() },
        rhs: if swapped { lhs } else { rhs },
        lhs_axes: vec![usize::from(!swapped)],
        rhs_axes: vec![usize::from(swapped)],
        output_axes: if swapped { vec![1, 0] } else { vec![0, 1] },
        dense: false,
    };
    let first_lhs = StackedTensorMap::pack(&[&base.lhs]).unwrap();
    let first_rhs = StackedTensorMap::pack(&[&base.rhs]).unwrap();
    let plan = ContractPlan::new(&first_lhs, &first_rhs, &base.spec()).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut other_workspace = plan.workspace().unwrap();
    for count in [1, 2, 17] {
        let members: Vec<_> = (0..count)
            .map(|i| Case {
                name: base.name,
                lhs: base
                    .lhs
                    .scale(D::entry(1.0 + i as f64 / 8.0, i as f64 / 16.0))
                    .unwrap(),
                rhs: base
                    .rhs
                    .scale(D::entry(1.0 - i as f64 / 32.0, 0.0))
                    .unwrap(),
                lhs_axes: base.lhs_axes.clone(),
                rhs_axes: base.rhs_axes.clone(),
                output_axes: base.output_axes.clone(),
                dense: false,
            })
            .collect();
        let left: Vec<_> = members.iter().map(|member| &member.lhs).collect();
        let right: Vec<_> = members.iter().map(|member| &member.rhs).collect();
        let lhs = StackedTensorMap::pack(&left).unwrap();
        let rhs = StackedTensorMap::pack(&right).unwrap();
        let result = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let eager: Vec<_> = members.iter().map(Case::host).collect();
        assert_eq!(
            result.signature(),
            StackedTensorMap::pack(&[&eager[0]]).unwrap().signature()
        );
        for (i, member) in members.iter().enumerate() {
            let oracle = fermionic_blas_contract_oracle_partitioned(
                member,
                if swapped { TwistRole::A } else { TwistRole::B },
                1,
                twist,
            );
            numerics::assert_nonzero_slices_close(
                member.name,
                result.member(i).unwrap().dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                member.terms(),
            );
            if swapped {
                // The public A-side oracle alone cannot check the selected
                // zero-copy dispatch, which exchanges operands and twists B.
                let selected = Case {
                    name: "selected signed swapped dispatch",
                    lhs: member.rhs.clone(),
                    rhs: member.lhs.clone(),
                    lhs_axes: vec![1],
                    rhs_axes: vec![0],
                    output_axes: vec![0, 1],
                    dense: false,
                };
                let selected_oracle =
                    fermionic_blas_contract_oracle_partitioned(&selected, TwistRole::B, 1, twist);
                numerics::assert_nonzero_slices_close(
                    selected.name,
                    result.member(i).unwrap().dense_data().unwrap(),
                    selected_oracle.dense_data().unwrap(),
                    selected.terms(),
                );
                let untwisted = fermionic_blas_contract_oracle_partitioned(
                    &selected,
                    TwistRole::None,
                    1,
                    twist,
                );
                let expected = selected_oracle.dense_data().unwrap();
                let scale = expected
                    .iter()
                    .map(|value| value.magnitude())
                    .fold(0.0, f64::max);
                let tolerance =
                    64.0 * (selected.terms().max(1) as f64).sqrt() * D::EPS * (1.0 + scale);
                assert!(
                    expected
                        .iter()
                        .zip(untwisted.dense_data().unwrap())
                        .any(|(&a, &b)| a.distance(b) > tolerance),
                    "{}: the selected B-side twist is vacuous for {}",
                    selected.name,
                    D::NAME,
                );
            }
            numerics::assert_nonzero_slices_close(
                member.name,
                result.member(i).unwrap().dense_data().unwrap(),
                eager[i].dense_data().unwrap(),
                member.terms(),
            );
        }
        let poisoned: Vec<_> = members.iter().map(poisoned_destination).collect();
        let refs: Vec<_> = poisoned.iter().collect();
        let mut dst = StackedTensorMap::pack(&refs).unwrap();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut other_workspace)
            .unwrap();
        for (i, expected) in eager.iter().enumerate() {
            numerics::assert_nonzero_slices_close(
                base.name,
                dst.member(i).unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap(),
                base.terms(),
            );
        }
        if count == 2 {
            let short_rhs = StackedTensorMap::pack(&[&members[0].rhs]).unwrap();
            let snapshot: Vec<Vec<D>> = (0..count)
                .map(|i| dst.member(i).unwrap().dense_data().unwrap().to_vec())
                .collect();
            assert!(plan
                .execute_into(&lhs, &short_rhs, &mut dst, &mut other_workspace)
                .is_err());
            for (i, before) in snapshot.iter().enumerate() {
                assert_eq!(dst.member(i).unwrap().dense_data().unwrap(), before);
            }
        }
    }
}

#[test]
fn mixed_signed_fermionic_members_match_literal_tensorkit_steps() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! case {
        ($rule:ty, $dtype:ty, $space:expr) => {{
            let space = $space;
            for swapped in [false, true] {
                check_mixed_signed_members(
                    &runtime,
                    &space,
                    swapped,
                    |tensor: &TensorMap<$rule, $dtype>, legs| {
                        tensor
                            .twist(legs, symtn::typed::Direction::Forward)
                            .unwrap()
                    },
                );
            }
        }};
    }
    case!(contract_cases::FermionU1, f64, fermion_u1());
    case!(
        contract_cases::FermionU1,
        symtn::typed::Complex64,
        fermion_u1()
    );
    case!(contract_cases::FermionSu2, f64, fermion_su2());
    case!(
        contract_cases::FermionSu2,
        symtn::typed::Complex64,
        fermion_su2()
    );
    use symtn::sector::{product_sector, FermionParityFusionRule, U1FusionRule, U1Irrep, Z2Irrep};
    let odd_only = || {
        GradedSpace::try_new(
            Arc::new(FermionParityFusionRule.product(U1FusionRule)),
            [(product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 2)],
        )
        .unwrap()
    };
    case!(contract_cases::FermionU1, f64, odd_only());
    case!(
        contract_cases::FermionU1,
        symtn::typed::Complex64,
        odd_only()
    );
}

#[test]
fn partitioned_output_uses_requested_codomain_rank() {
    let runtime = Runtime::builder().build().unwrap();
    let case = u1_reordered::<f64>(&runtime);
    let lhs = StackedTensorMap::pack(&[&case.lhs]).unwrap();
    let rhs = StackedTensorMap::pack(&[&case.rhs]).unwrap();
    let mut admitted = 0;
    for split in [0, 1, 3] {
        let (codomain, domain) = case.output_axes.split_at(split);
        let spec = ContractSpec {
            lhs: &case.lhs_axes,
            rhs: &case.rhs_axes,
            codomain,
            domain,
        };
        let Ok(plan) = ContractPlan::new(&lhs, &rhs, &spec) else {
            continue;
        };
        let expected = case.lhs.contract(&case.rhs, &spec).unwrap();
        let actual = plan
            .execute(&lhs, &rhs, &mut plan.workspace().unwrap())
            .unwrap()
            .member(0)
            .unwrap();
        assert_eq!(actual.codomain_rank(), split);
        numerics::assert_nonzero_slices_close(
            case.name,
            actual.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            case.terms(),
        );
        admitted += 1;
    }
    assert!(admitted > 0, "no non-default output split was admitted");
    let malformed = ContractSpec {
        lhs: &case.lhs_axes,
        rhs: &case.rhs_axes,
        codomain: &[0, 0],
        domain: &[2],
    };
    assert!(
        matches!(ContractPlan::new(&lhs, &rhs, &malformed), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidPermutation { rank: 3, .. }))
    );
}

/// `case` with each member's operands rescaled: `lhs` by `lhs_scale`, `rhs`
/// by a member-dependent real factor.
fn rescaled_members<R, D>(case: &Case<R, D>, count: usize, lhs_scale: f64) -> Vec<Case<R, D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    (0..count)
        .map(|i| Case {
            name: case.name,
            lhs: case
                .lhs
                .scale(D::entry(lhs_scale - i as f64 / 64.0, (i + 1) as f64 / 32.0))
                .unwrap(),
            rhs: case
                .rhs
                .scale(D::entry(1.25 + i as f64 / 16.0, 0.0))
                .unwrap(),
            lhs_axes: case.lhs_axes.clone(),
            rhs_axes: case.rhs_axes.clone(),
            output_axes: case.output_axes.clone(),
            dense: case.dense,
        })
        .collect()
}

/// Changed-input reuse of the workspace-owned output (#1732, #2123): over
/// the B sequences 1, 2, 17, 1 and 4, 2, 4 on one workspace, each call
/// executes stale inputs, then a rejected call (its buffer becomes the
/// unobservable spare), then fresh inputs into that same buffer, and finally
/// `execute_into` a NaN-poisoned destination. Every member equals TensorKit
/// `blas_contract!` on its own fresh operands, so a block the reused buffer
/// kept from the stale call, including an inactive block, would differ.
fn check_reused_output<R, D>(case: &Case<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: Payload,
{
    let spec = case.spec();
    let one = |tensor: &TensorMap<R, D>| StackedTensorMap::pack(&[tensor]).unwrap();
    let plan = ContractPlan::new(&one(&case.lhs), &one(&case.rhs), &spec).unwrap();
    let stack = |members: &[Case<R, D>], pick: fn(&Case<R, D>) -> &TensorMap<R, D>| {
        StackedTensorMap::pack(&members.iter().map(pick).collect::<Vec<_>>()).unwrap()
    };
    for sequence in [&[1usize, 2, 17, 1][..], &[4, 2, 4]] {
        let mut workspace = plan.workspace().unwrap();
        for (call, &count) in sequence.iter().enumerate() {
            let what = format!("{} {} seq={sequence:?} B={count}", case.name, D::NAME);
            let stale = rescaled_members(case, count, 2.0 + call as f64);
            plan.execute(
                &stack(&stale, |m| &m.lhs),
                &stack(&stale, |m| &m.rhs),
                &mut workspace,
            )
            .unwrap();
            let longer = rescaled_members(case, count + 1, 1.0);
            assert!(plan
                .execute(
                    &stack(&stale, |m| &m.lhs),
                    &stack(&longer, |m| &m.rhs),
                    &mut workspace
                )
                .is_err());
            assert!(workspace.take_output().is_none(), "{what}");
            let fresh = rescaled_members(case, count, -0.75 - call as f64 / 8.0);
            let (lhs, rhs) = (stack(&fresh, |m| &m.lhs), stack(&fresh, |m| &m.rhs));
            let returned = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            let oracles: Vec<_> = fresh.iter().map(blas_contract_oracle).collect();
            for (i, oracle) in oracles.iter().enumerate() {
                numerics::assert_nonzero_slices_close(
                    &format!("{what} execute member {i}"),
                    returned.member(i).unwrap().dense_data().unwrap(),
                    oracle.dense_data().unwrap(),
                    case.terms(),
                );
            }
            let poison: Vec<_> = fresh.iter().map(poisoned_destination).collect();
            let mut dst = StackedTensorMap::pack(&poison).unwrap();
            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                .unwrap();
            for (i, oracle) in oracles.iter().enumerate() {
                numerics::assert_nonzero_slices_close(
                    &format!("{what} execute_into member {i}"),
                    dst.member(i).unwrap().dense_data().unwrap(),
                    oracle.dense_data().unwrap(),
                    case.terms(),
                );
            }
        }
    }
}

#[test]
fn reused_output_follows_changed_inputs_on_every_route() {
    // Core with an inactive block, CopyC (U(1) both orientations, one over
    // an inactive core block, and SU(2)), DynamicTree with an identity output
    // over an inactive block and with source and output transforms.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    fn cases<D: Payload>(runtime: &Runtime) {
        for case in u1_inactive_cases::<D>(runtime) {
            check_reused_output(&case);
        }
        for (case, _) in contract_cases::copy_c_probes::<_, D>(runtime, &u1_non_self_dual()) {
            if matches!(case.name, "C1p" | "C2p") {
                check_reused_output(&case);
            }
        }
        for (case, _) in contract_cases::copy_c_probes::<_, D>(runtime, &su2()) {
            if case.name == "C1p" {
                check_reused_output(&case);
            }
        }
        check_reused_output(&u1_reordered::<D>(runtime));
    }
    cases::<f64>(&runtime);
    cases::<symtn::typed::Complex64>(&runtime);
}

fn reject_braiding<const ANYONIC: bool>(runtime: &Runtime) -> bool {
    let leg =
        GradedSpace::try_new(Arc::new(RealBraidingProbe::<ANYONIC>), [(ProbeSector, 2)]).unwrap();
    let tensor = TensorMap::<_, f64>::zeros(runtime, [&leg], [&leg]).unwrap();
    let stack = StackedTensorMap::pack(&[tensor]).unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    matches!(ContractPlan::new(&stack, &stack, &spec), Err(Error::Operation(error)) if matches!(*error, symtn::typed::OperationError::UnsupportedTensorContractScope { message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED }))
}
