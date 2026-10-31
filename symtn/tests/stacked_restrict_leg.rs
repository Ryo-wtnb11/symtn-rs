//! `StackedTensorMap::restrict_leg` on Host.
//!
//! The oracle is eager `TensorMap::restrict_leg` applied to each member: the
//! stacked result must equal it bit for bit, with the same space, and reject
//! the same inputs with the same errors. A multi-axis set is additionally
//! checked against sequential one-axis restrictions of each member.

use num_complex::Complex64;
use std::sync::Arc;

use symtn::typed::{GradedSpace, LegSelection, StackedTensorMap, TensorMap};

use crate::stacked_fixtures as fixtures;

use crate::fixtures as shared_fixtures;

use shared_fixtures::host_runtime;

macro_rules! restrict_members {
    ($engine:ident, $label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let dual = a.try_dual().unwrap();
        let runtime = host_runtime();
        restrict_members!($engine, @dtype $label, &runtime, &a, &dual, f64);
        restrict_members!($engine, @dtype $label, &runtime, &a, &dual, Complex64);
    }};
    ($engine:ident, @dtype $label:expr, $runtime:expr, $a:expr, $dual:expr, $d:ty) => {{
        for count in [1usize, 3] {
            let members = mixed_members!($engine, $runtime, $a, $dual, $d, count);
            let stack = StackedTensorMap::pack(&members).unwrap();
            for (axis, parent) in [$a, $dual, $a, $dual].into_iter().enumerate() {
                for (choice, selection) in selections(parent).iter().enumerate() {
                    let label = format!(
                        "{} {} B={count} axis {axis} selection {choice}",
                        $label,
                        stringify!($d)
                    );
                    let restricted = stack.restrict_leg($engine, &[(axis, selection)]).unwrap();
                    assert_eq!(restricted.len(), count, "{label}");
                    for (i, member) in members.iter().enumerate() {
                        let expected = member.restrict_leg($engine, &[(axis, selection)]).unwrap();
                        let actual = restricted.member($engine, i).unwrap();
                        assert!(
                            actual.structure_signature() == expected.structure_signature(),
                            "{label}: member {i} space"
                        );
                        assert!(
                            *restricted.signature() == expected.structure_signature(),
                            "{label}: stack signature"
                        );
                        fixtures::assert_bit_exact(actual.dense_data().unwrap(), expected.dense_data().unwrap(), &label);
                    }
                }
            }
        }
    }};
}

#[test]
fn restrict_leg_equals_eager_per_member_bitwise() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for_each_symmetry!(engine, restrict_members);
}

/// The multi-axis sets: all four legs in a shuffled order, three legs, and
/// two legs, each leg with its own selection; and two non-contiguous
/// selections next to a contiguous one.
fn multi_axis_sets<'s, R>(
    on_a: &'s [LegSelection<R>],
    on_dual: &'s [LegSelection<R>],
) -> Vec<Vec<(usize, &'s LegSelection<R>)>> {
    vec![
        vec![
            (2, &on_a[1]),
            (0, &on_a[0]),
            (3, &on_dual[0]),
            (1, &on_dual[1]),
        ],
        vec![(1, &on_dual[0]), (2, &on_a[0]), (0, &on_a[1])],
        vec![(3, &on_dual[1]), (0, &on_a[0])],
        vec![(1, &on_dual[2]), (2, &on_a[2]), (3, &on_dual[0])],
    ]
}

macro_rules! multi_axis_members {
    ($engine:ident, $label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let dual = a.try_dual().unwrap();
        let runtime = host_runtime();
        multi_axis_members!($engine, @dtype $label, &runtime, &a, &dual, f64);
        multi_axis_members!($engine, @dtype $label, &runtime, &a, &dual, Complex64);
    }};
    ($engine:ident, @dtype $label:expr, $runtime:expr, $a:expr, $dual:expr, $d:ty) => {{
        let members = mixed_members!($engine, $runtime, $a, $dual, $d, 3);
        let stack = StackedTensorMap::pack(&members).unwrap();
        let (on_a, on_dual) = (selections($a), selections($dual));
        for (choice, set) in multi_axis_sets(&on_a, &on_dual).iter().enumerate() {
            let label = format!("{} {} set {choice}", $label, stringify!($d));
            let restricted = stack.restrict_leg($engine, set).unwrap();
            for (i, member) in members.iter().enumerate() {
                // Oracle: one-axis restrictions applied in turn.
                let sequential = set.iter().fold(member.clone(), |tensor, &pair| {
                    tensor.restrict_leg($engine, &[pair]).unwrap()
                });
                let eager = member.restrict_leg($engine, set).unwrap();
                assert!(
                    eager.structure_signature() == sequential.structure_signature(),
                    "{label}: member {i} eager space"
                );
                fixtures::assert_bit_exact(eager.dense_data().unwrap(), sequential.dense_data().unwrap(), &label);

                // A lazy adjoint is read in place, as `materialize` then
                // restrict would.
                let lazy = member.adjoint($engine).unwrap();
                fixtures::assert_bit_exact(
                    lazy.restrict_leg($engine, set).unwrap().dense_data().unwrap(),
                    lazy.materialize($engine).unwrap().restrict_leg($engine, set).unwrap().dense_data().unwrap(),
                    &label,
                );

                let actual = restricted.member($engine, i).unwrap();
                assert!(
                    actual.structure_signature() == sequential.structure_signature(),
                    "{label}: member {i} stacked space"
                );
                fixtures::assert_bit_exact(actual.dense_data().unwrap(), sequential.dense_data().unwrap(), &label);
            }
        }
    }};
}

#[test]
fn multi_axis_restrict_leg_equals_sequential_one_axis_restrictions_bitwise() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for_each_symmetry!(engine, multi_axis_members);
}

macro_rules! malformed {
    ($engine:ident, $label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let dual = a.try_dual().unwrap();
        let other = leg(1);
        let runtime = host_runtime();
        let members = mixed_members!($engine, &runtime, &a, &dual, f64, 2);
        let stack = StackedTensorMap::pack(&members).unwrap();
        let on_a = &selections(&a)[0];
        let on_dual = &selections(&dual)[0];
        let on_other = &selections(&other)[0];
        // Out of range, the dual of the selected leg, and a leg of another
        // degeneracy: each is eager's error, word for word.
        for (axis, selection) in [(4, on_a), (0, on_dual), (1, on_a), (2, on_other)] {
            let expected = members[0]
                .restrict_leg($engine, &[(axis, selection)])
                .err()
                .unwrap();
            let actual = stack
                .restrict_leg($engine, &[(axis, selection)])
                .err()
                .unwrap();
            assert_eq!(
                format!("{actual:?}"),
                format!("{expected:?}"),
                "{} axis {axis}",
                $label
            );
        }
        // An empty set and a repeated axis: eager's errors, word for word.
        for set in [vec![], vec![(0, on_a), (2, on_a), (0, on_a)]] {
            let expected = members[0].restrict_leg($engine, &set).err().unwrap();
            let actual = stack.restrict_leg($engine, &set).err().unwrap();
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "{}", $label);
        }
        assert!(
            format!("{:?}", stack.restrict_leg($engine, &[]).err().unwrap())
                .contains("at least one")
        );
        assert!(format!(
            "{:?}",
            stack
                .restrict_leg($engine, &[(0, on_a), (2, on_a), (0, on_a)])
                .err()
                .unwrap()
        )
        .contains("axis 0 appears twice"));
    }};
}

#[test]
fn restrict_leg_rejects_what_eager_rejects() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for_each_symmetry!(engine, malformed);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn a_selection_of_another_rule_instance_is_a_rule_mismatch() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = host_runtime();
    let leg = |rank: usize, trivial: Vec<i64>| {
        let rule = Arc::new(
            symtn::sector::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(rank).unwrap())
                .unwrap(),
        );
        GradedSpace::try_new(rule, [(trivial, 2)]).unwrap()
    };
    let su3 = leg(3, vec![0, 0]);
    let su4 = leg(4, vec![0, 0, 0]);
    let member = TensorMap::<_, f64>::rand_with_seed(
        engine,
        &runtime,
        [&su3],
        [&su3],
        0x9E37_79B9_7F4A_7C15,
    )
    .unwrap();
    let stack = StackedTensorMap::pack(&[&member, &member]).unwrap();
    let foreign = LegSelection::try_new(&su4, [(vec![0i64, 0, 0], 0..1)]).unwrap();
    let expected = member.restrict_leg(engine, &[(0, &foreign)]).err().unwrap();
    let actual = stack.restrict_leg(engine, &[(0, &foreign)]).err().unwrap();
    assert!(
        matches!(
            expected,
            symtn::typed::GenericTensorError::Facade(symtn::typed::Error::RuleMismatch)
        ),
        "{expected:?}"
    );
    assert!(
        matches!(
            actual,
            symtn::typed::GenericTensorError::Facade(symtn::typed::Error::RuleMismatch)
        ),
        "{actual:?}"
    );
}

#[test]
fn restrict_leg_of_a_blockless_structure_is_empty() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let q = fixtures::U1Irrep::new;
    let charged = GradedSpace::try_new(Arc::new(fixtures::U1FusionRule), [(q(1), 2)]).unwrap();
    let neutral = GradedSpace::try_new(Arc::new(fixtures::U1FusionRule), [(q(0), 2)]).unwrap();
    let empty = TensorMap::<_, f64>::zeros(engine, &runtime, [&charged], [&neutral]).unwrap();
    let stack = StackedTensorMap::pack(&[&empty, &empty, &empty]).unwrap();
    let selection = LegSelection::try_new(&charged, [(q(1), 1..2)]).unwrap();
    let restricted = stack.restrict_leg(engine, &[(0, &selection)]).unwrap();
    let expected = empty.restrict_leg(engine, &[(0, &selection)]).unwrap();
    assert_eq!(restricted.len(), 3);
    assert!(*restricted.signature() == expected.structure_signature());
    assert!(restricted
        .member(engine, 2)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}
