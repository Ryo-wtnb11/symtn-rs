//! Host gates of `ComposePlan` and its `ContractWorkspace` (#1639).
//!
//! Per member both public call shapes must equal eager `compose` of the same
//! placement and the tree-keyed `mul!` oracle of `prepared/mod.rs`,
//! over U(1), SU(2) and fZ2xU(1), f64 and c64, B in {1, 2, 17}, with inactive
//! destination blocks. `execute_into` must give the same result over a
//! destination prefilled with NaN or garbage.
//!
//! #1775: the composition is its own kind, not a contraction inferred from
//! its axes. A fermionic composition over a dual leg differs from the
//! contraction over the same legs by the supertrace sign, and composition
//! admits every braiding style while `ContractPlan` rejects non-symmetric
//! ones.

mod common;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;

#[path = "braiding_probe/mod.rs"]
mod braiding_probe;

#[path = "support/report.rs"]
mod report;
use report::Report;

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use std::sync::Arc;

use braiding_probe::{ProbeSector, RealBraidingProbe};
use symtn::sector::{
    product_sector, CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec, U1Irrep,
    Z2Irrep,
};
use symtn::typed::Error;
use symtn::typed::{
    ComposePlan, ContractPlan, ContractSpec, GradedSpace, Runtime, SignatureField,
    StackedTensorMap, TensorMap,
};

use common::Payload;
use prepared::{
    block_matrix, compose_oracle, filled, fz2u1_legs, members, su2_legs, u1_legs, Fz2U1Rule,
};

const MEMBER_COUNTS: [usize; 3] = [1, 2, 17];

#[test]
fn one_plan_has_isolated_workspaces_across_member_counts() {
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let stacks = |count, seed| {
        (
            StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], count, seed))
                .unwrap(),
            StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], count, seed + 1))
                .unwrap(),
        )
    };
    let (lhs2, rhs2) = stacks(2, 10);
    let (lhs17, rhs17) = stacks(17, 20);
    let plan = ComposePlan::new(&lhs2, &rhs2).unwrap();
    let mut first = plan.workspace().unwrap();
    let mut second = plan.workspace().unwrap();

    let first_values = plan
        .execute(&lhs2, &rhs2, &mut first)
        .unwrap()
        .member(0)
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let first_bytes = first.retained_bytes();
    assert_eq!(plan.execute(&lhs17, &rhs17, &mut second).unwrap().len(), 17);
    assert_eq!(
        plan.execute(&lhs2, &rhs2, &mut first)
            .unwrap()
            .member(0)
            .unwrap()
            .dense_data()
            .unwrap(),
        first_values
    );
    assert_eq!(first.retained_bytes(), first_bytes);
    assert!(second.retained_bytes() > first.retained_bytes());
}

#[test]
fn one_plan_supports_concurrent_independent_workspaces() {
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 2, 30)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 31)).unwrap();
    let plan = std::sync::Arc::new(ComposePlan::new(&lhs, &rhs).unwrap());
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..2 {
            let plan = std::sync::Arc::clone(&plan);
            let (lhs, rhs) = (&lhs, &rhs);
            threads.push(scope.spawn(move || {
                let mut workspace = plan.workspace().unwrap();
                plan.execute(lhs, rhs, &mut workspace)
                    .unwrap()
                    .member(0)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .to_vec()
            }));
        }
        let first = threads.remove(0).join().unwrap();
        let second = threads.remove(0).join().unwrap();
        assert_eq!(first, second);
        assert!(first.iter().any(|&value| value != 0.0));
    });
}

#[test]
fn foreign_workspace_is_rejected_before_destination_write() {
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 2, 50)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 51)).unwrap();
    let mut dst =
        StackedTensorMap::pack(&filled::<_, f64>(&runtime, &[&v, &v], &[&v], 2, f64::NAN)).unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let foreign = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut workspace = foreign.workspace().unwrap();
    assert!(plan
        .execute_into(&lhs, &rhs, &mut dst, &mut workspace)
        .is_err());
    assert!(dst
        .member(0)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.is_nan()));
}

fn equivalence<R, D>(label: &str, (v, w): (GradedSpace<R>, GradedSpace<R>))
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
    D: Payload,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let runtime = Runtime::builder().build().unwrap();
    for count in MEMBER_COUNTS {
        let label = format!("{label} {} B={count}", D::NAME);
        let a = members::<R, D>(&runtime, &[&v, &v], &[&w], count, 1);
        let b = members::<R, D>(&runtime, &[&w], &[&v], count, 2);
        let lhs = StackedTensorMap::pack(&a).unwrap();
        let rhs = StackedTensorMap::pack(&b).unwrap();
        let plan = ComposePlan::new(&lhs, &rhs).unwrap();
        let mut ws = plan.workspace().unwrap();

        // `terms` = len(A), an upper bound on the inner dimension of a block.
        let terms = a[0].dense_data().unwrap().len();
        let mut oracles = Vec::new();
        for (x, y) in a.iter().zip(&b) {
            let eager = x.compose(y).unwrap();
            let (oracle, unreached) = compose_oracle(x, y, &eager);
            assert!(unreached > 0, "{label}: fixture must have inactive blocks");
            assert!(
                eager.subblock_count() > unreached + 1,
                "{label}: several active blocks"
            );
            numerics::assert_nonzero_slices_close(
                &format!("{label}: eager"),
                eager.dense_data().unwrap(),
                &oracle,
                terms,
            );
            oracles.push(oracle);
        }

        let output = plan.execute(&lhs, &rhs, &mut ws).unwrap();
        assert_eq!(output.len(), count);
        assert!(*output.signature() == a[0].compose(&b[0]).unwrap().structure_signature());
        for (index, oracle) in oracles.iter().enumerate() {
            let member = output.member(index).unwrap();
            numerics::assert_nonzero_slices_close(
                &format!("{label}: member {index}"),
                member.dense_data().unwrap(),
                oracle,
                terms,
            );
        }

        for (fill, name) in [
            (D::entry(f64::NAN, f64::NAN), "NaN"),
            (D::entry(-3.0e17, 7.5), "garbage"),
        ] {
            let mut dst =
                StackedTensorMap::pack(&filled::<R, D>(&runtime, &[&v, &v], &[&v], count, fill))
                    .unwrap();
            assert!(*dst.signature() == *plan.output_signature());
            plan.execute_into(&lhs, &rhs, &mut dst, &mut ws).unwrap();
            for (index, oracle) in oracles.iter().enumerate() {
                let member = dst.member(index).unwrap();
                numerics::assert_nonzero_slices_close(
                    &format!("{label}: execute_into over {name}, member {index}"),
                    member.dense_data().unwrap(),
                    oracle,
                    terms,
                );
            }
        }
    }
}

#[test]
fn prepared_compose_equals_eager_and_the_tree_oracle_per_member() {
    equivalence::<_, f64>("U1", u1_legs());
    equivalence::<_, Complex64>("U1", u1_legs());
    equivalence::<_, f64>("SU2", su2_legs());
    equivalence::<_, Complex64>("SU2", su2_legs());
    equivalence::<_, f64>("fZ2xU1", fz2u1_legs());
    equivalence::<_, Complex64>("fZ2xU1", fz2u1_legs());
}

#[test]
fn warm_replay_retains_the_same_bytes_and_a_new_member_count_resizes() {
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let stacks = |count| {
        (
            StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 1))
                .unwrap(),
            StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], count, 2)).unwrap(),
        )
    };
    let (lhs, rhs) = stacks(4);
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut ws = plan.workspace().unwrap();
    assert_eq!(ws.retained_bytes(), 0, "nothing is allocated before a call");
    plan.execute(&lhs, &rhs, &mut ws).unwrap();
    let cold = ws.retained_bytes();
    assert!(cold > 0);
    for _ in 0..3 {
        plan.execute(&lhs, &rhs, &mut ws).unwrap();
        assert_eq!(
            ws.retained_bytes(),
            cold,
            "a warm replay retains nothing new"
        );
    }
    let (wide_lhs, wide_rhs) = stacks(9);
    plan.execute(&wide_lhs, &wide_rhs, &mut ws).unwrap();
    assert!(ws.retained_bytes() > cold, "a larger B resizes the output");

    // `execute_into` across a change of B on the same workspace: the job list
    // is rebuilt for each B and the result stays the oracle's.
    for count in [4, 9, 4] {
        let a = members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 1);
        let b = members::<_, f64>(&runtime, &[&w], &[&v], count, 2);
        let (lhs, rhs) = (
            StackedTensorMap::pack(&a).unwrap(),
            StackedTensorMap::pack(&b).unwrap(),
        );
        let mut dst = StackedTensorMap::pack(&filled::<_, f64>(
            &runtime,
            &[&v, &v],
            &[&v],
            count,
            f64::NAN,
        ))
        .unwrap();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut ws).unwrap();
        for (index, (x, y)) in a.iter().zip(&b).enumerate() {
            let (oracle, _) = compose_oracle(x, y, &x.compose(y).unwrap());
            numerics::assert_nonzero_slices_close(
                &format!("execute_into at B={count}, member {index}"),
                dst.member(index).unwrap().dense_data().unwrap(),
                &oracle,
                x.dense_data().unwrap().len(),
            );
        }
    }
    plan.execute(&wide_lhs, &wide_rhs, &mut ws).unwrap();

    let output = ws.take_output().unwrap();
    assert_eq!(output.len(), 9);
    assert!(
        ws.retained_bytes() < cold,
        "the moved output is no longer retained"
    );
}

#[test]
fn mismatched_stacks_are_typed_errors_before_any_work() {
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let lhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], 3, 1)).unwrap();
    let rhs = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 3, 2)).unwrap();
    let short = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], 2, 2)).unwrap();
    let foreign =
        StackedTensorMap::pack(&members::<_, f64>(&other_runtime, &[&w], &[&v], 3, 2)).unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut ws = plan.workspace().unwrap();

    assert_eq!(
        ComposePlan::new(&lhs, &foreign).err(),
        Some(Error::RuntimeMismatch)
    );
    assert_eq!(
        plan.execute(&rhs, &lhs, &mut ws).err(),
        Some(Error::BatchSignatureMismatch {
            member: None,
            field: SignatureField::HomSpace
        })
    );
    assert_eq!(
        plan.execute(&lhs, &foreign, &mut ws).err(),
        Some(Error::BatchSignatureMismatch {
            member: None,
            field: SignatureField::Runtime
        })
    );
    assert!(plan
        .execute(&lhs, &short, &mut ws)
        .err()
        .unwrap()
        .report()
        .contains("operand stacks have different member counts"));
    assert_eq!(ws.retained_bytes(), 0, "a rejected call allocates nothing");

    let mut wrong_dst =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&v], 3, 3)).unwrap();
    assert!(matches!(
        plan.execute_into(&lhs, &rhs, &mut wrong_dst, &mut ws),
        Err(Error::BatchSignatureMismatch { member: None, .. })
    ));
    let mut short_dst =
        StackedTensorMap::pack(&filled::<_, f64>(&runtime, &[&v, &v], &[&v], 2, 1.0)).unwrap();
    assert!(plan
        .execute_into(&lhs, &rhs, &mut short_dst, &mut ws)
        .err()
        .unwrap()
        .report()
        .contains("destination member count differs"));
    assert_eq!(
        short_dst.member(0).unwrap().dense_data().unwrap()[0],
        1.0,
        "nothing was written"
    );
}

/// Every member of `output` against the eager composition and the tree-keyed
/// `mul!` oracle; returns the eager results.
fn assert_members_compose<R, D>(
    label: &str,
    output: &StackedTensorMap<R, D>,
    a: &[TensorMap<R, D>],
    b: &[TensorMap<R, D>],
) -> Vec<TensorMap<R, D>>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    R::Sector: Debug,
    D: Payload,
{
    assert_eq!(output.len(), a.len());
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(index, (x, y))| {
            let eager = x.compose(y).unwrap();
            let (oracle, _) = compose_oracle(x, y, &eager);
            let terms = x.dense_data().unwrap().len();
            let member = output.member(index).unwrap();
            numerics::assert_nonzero_slices_close(
                &format!("{label}: eager, member {index}"),
                eager.dense_data().unwrap(),
                &oracle,
                terms,
            );
            numerics::assert_nonzero_slices_close(
                &format!("{label}: plan, member {index}"),
                member.dense_data().unwrap(),
                &oracle,
                terms,
            );
            eager
        })
        .collect()
}

#[test]
fn fermionic_composition_is_not_the_contraction_over_the_same_legs() {
    // What: `V <- W'` composed with `W' <- V` over fZ2 x U(1), whose rhs
    // codomain is a dual leg. `ComposePlan` is eager `compose` and TensorKit
    // `mul!` (no twist); `ContractPlan` over the same legs and output is
    // eager `contract`, which carries the supertrace sign: every block is
    // the composition's up to a sign, and the odd blocks flip.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(Fz2U1Rule::new(
        symtn::sector::FermionParityFusionRule,
        symtn::sector::U1FusionRule,
    ));
    let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
    let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
    let v =
        GradedSpace::try_new(Arc::clone(&rule), [(even(0), 2), (odd(1), 2), (odd(-1), 1)]).unwrap();
    let wd = GradedSpace::try_new(rule, [(even(0), 2), (odd(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    for count in MEMBER_COUNTS {
        let label = format!("fZ2xU1 dual leg B={count}");
        let a = members::<_, f64>(&runtime, &[&v], &[&wd], count, 7);
        let b = members::<_, f64>(&runtime, &[&wd], &[&v], count, 8);
        let (lhs, rhs) = (
            StackedTensorMap::pack(&a).unwrap(),
            StackedTensorMap::pack(&b).unwrap(),
        );
        let compose = ComposePlan::new(&lhs, &rhs).unwrap();
        let contract = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
        assert!(*compose.output_signature() == *contract.output_signature());
        let mut compose_ws = compose.workspace().unwrap();
        let mut contract_ws = contract.workspace().unwrap();
        let composed = compose.execute(&lhs, &rhs, &mut compose_ws).unwrap();
        let eager = assert_members_compose(&label, composed, &a, &b);
        let contracted = contract.execute(&lhs, &rhs, &mut contract_ws).unwrap();
        for (index, (x, y)) in a.iter().zip(&b).enumerate() {
            let mut flipped = 0;
            let expected = x.contract(y, &spec).unwrap();
            let member = contracted.member(index).unwrap();
            numerics::assert_nonzero_slices_close(
                &format!("{label}: contract plan, member {index}"),
                member.dense_data().unwrap(),
                expected.dense_data().unwrap(),
                x.dense_data().unwrap().len(),
            );
            for block in 0..eager[index].subblock_count() {
                let (_, _, _, _, composed) = block_matrix(&eager[index], block);
                let (_, _, _, _, contracted) = block_matrix(&member, block);
                if composed == contracted {
                    continue;
                }
                assert!(
                    composed.iter().zip(&contracted).all(|(c, t)| *c == -*t),
                    "{label}: member {index} block {block} differs by more than a sign"
                );
                flipped += 1;
            }
            assert!(
                flipped >= 1,
                "{label}: member {index}: the supertrace sign flips an odd block"
            );
        }
    }
}

fn non_symmetric_composition<const ANYONIC: bool>() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(RealBraidingProbe::<ANYONIC>);
    let v = GradedSpace::try_new(Arc::clone(&rule), [(ProbeSector, 2)]).unwrap();
    let w = GradedSpace::try_new(rule, [(ProbeSector, 3)]).unwrap();
    for count in [1, 2] {
        let label = format!("anyonic={ANYONIC} B={count}");
        let a = members::<_, f64>(&runtime, &[&v, &v], &[&w], count, 3);
        let b = members::<_, f64>(&runtime, &[&w], &[&v], count, 4);
        let (lhs, rhs) = (
            StackedTensorMap::pack(&a).unwrap(),
            StackedTensorMap::pack(&b).unwrap(),
        );
        let plan = ComposePlan::new(&lhs, &rhs).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        assert_members_compose(&label, output, &a, &b);
        let mut dst = StackedTensorMap::pack(&filled::<_, f64>(
            &runtime,
            &[&v, &v],
            &[&v],
            count,
            f64::NAN,
        ))
        .unwrap();
        plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
            .unwrap();
        assert_members_compose(&format!("{label} execute_into"), &dst, &a, &b);
        let spec = ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2],
        };
        assert!(
            matches!(
                ContractPlan::new(&lhs, &rhs, &spec),
                Err(Error::Operation(error)) if matches!(
                    *error,
                    symtn::typed::OperationError::UnsupportedTensorContractScope {
                        message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
                    }
                )
            ),
            "{label}"
        );
    }
}

/// F2 (#1775): composition admits every braiding style (TensorKit `mul!`
/// checks only the spaces), and no shared planning or admission step rejects
/// it, while `ContractPlan::new` rejects the same geometry. Fibonacci, the
/// built-in anyonic rule, has complex symbols (`Scalar = Complex64`), which
/// stacks do not take (`R: MultiplicityFreeRigidSymbols<Scalar = f64>`), so
/// the real one-sector probes stand in for `Anyonic` and `NoBraiding`.
#[test]
fn non_symmetric_braiding_composes_and_contract_plan_rejects_it() {
    non_symmetric_composition::<true>();
    non_symmetric_composition::<false>();
}
