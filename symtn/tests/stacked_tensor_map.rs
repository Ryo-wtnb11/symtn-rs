//! `StackedTensorMap` on Host.

use num_complex::Complex64;
use std::sync::Arc;

use symtn::typed::Error;
use symtn::typed::{
    BatchMemberRepresentation, GradedSpace, SectorSpectrum, SignatureField, StackedTensorMap,
    TensorMap,
};

use crate::stacked_fixtures as fixtures;

use crate::fixtures as shared_fixtures;

use shared_fixtures::host_runtime;

macro_rules! round_trip {
    ($engine:ident, $label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let runtime = host_runtime();
        round_trip!($engine, @dtype $label, &runtime, &a, f64);
        round_trip!($engine, @dtype $label, &runtime, &a, Complex64);
    }};
    ($engine:ident, @dtype $label:expr, $runtime:expr, $a:expr, $d:ty) => {{
        let label = format!("{} {}", $label, stringify!($d));
        let members = members!($engine, $runtime, $a, $d, 3);
        let stack = StackedTensorMap::pack(&members).unwrap();
        assert_eq!(stack.len(), 3);
        assert!(*stack.signature() == members[0].structure_signature());
        for (index, member) in members.iter().enumerate() {
            let unpacked = stack.member($engine, index).unwrap();
            assert!(unpacked.structure_signature() == member.structure_signature());
            fixtures::assert_bit_exact(unpacked.dense_data().unwrap(), member.dense_data().unwrap(), &label);
        }
        assert_eq!(
            stack.member($engine, 3).err(),
            Some(Error::BatchMemberOutOfRange { member: 3, len: 3 })
        );
    }};
}

#[test]
fn member_of_pack_is_bit_exact() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for_each_symmetry!(engine, round_trip);
}

fn assert_drift(result: Result<StackedTensorMap<impl Sized, f64>, Error>, field: SignatureField) {
    match result {
        Err(error) => assert_eq!(
            error,
            Error::BatchSignatureMismatch {
                member: Some(2),
                field
            }
        ),
        Ok(_) => panic!("drift must be rejected"),
    }
}

macro_rules! space_drift {
    ($engine:ident, $label:expr, $leg:expr) => {{
        let leg = $leg;
        let (a, degeneracy) = (leg(0), leg(1));
        let dual = a.try_dual().unwrap();
        let runtime = host_runtime();
        let base = members!($engine, &runtime, &a, f64, 2);
        for other in [&degeneracy, &dual] {
            let drifted = TensorMap::<_, f64>::zeros($engine, &runtime, [&a, other], [&a]).unwrap();
            assert_drift(
                StackedTensorMap::pack(&[&base[0], &base[1], &drifted]),
                SignatureField::HomSpace,
            );
        }
    }};
}

#[test]
fn space_drift_names_the_member_and_field() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for_each_symmetry!(engine, space_drift);
}

#[test]
fn runtime_drift_names_the_member_and_field() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let leg = GradedSpace::try_new(
        Arc::new(fixtures::U1FusionRule),
        [
            (fixtures::U1Irrep::new(0), 2),
            (fixtures::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let (first, second) = (host_runtime(), host_runtime());
    let base = members!(engine, &first, &leg, f64, 2);
    let drifted = members!(engine, &second, &leg, f64, 1);
    assert_drift(
        StackedTensorMap::pack(&[&base[0], &base[1], &drifted[0]]),
        SignatureField::Runtime,
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn rule_instance_drift_names_the_member_and_field() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = host_runtime();
    let mut tensor = |rank: usize, trivial: Vec<i64>| {
        let rule = Arc::new(
            symtn::sector::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(rank).unwrap())
                .unwrap(),
        );
        let leg = GradedSpace::try_new(rule, [(trivial, 2)]).unwrap();
        TensorMap::<_, f64>::zeros(engine, &runtime, [&leg], [&leg]).unwrap()
    };
    let su3 = tensor(3, vec![0, 0]);
    assert_drift(
        StackedTensorMap::pack(&[&su3, &su3, &tensor(4, vec![0, 0, 0])]),
        SignatureField::Rule,
    );
}

#[test]
fn lazy_adjoint_and_compact_diagonal_members_are_rejected() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let q = fixtures::U1Irrep::new;
    let bond =
        GradedSpace::try_new(Arc::new(fixtures::U1FusionRule), [(q(0), 2), (q(1), 1)]).unwrap();
    let square = TensorMap::<_, f64>::rand_with_seed(
        engine,
        &runtime,
        [&bond],
        [&bond],
        0x9E37_79B9_7F4A_7C15,
    )
    .unwrap();
    let lazy = square.adjoint(engine).unwrap();
    let diagonal = TensorMap::<_, f64>::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: q(0),
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: q(1),
                values: vec![3.0],
            },
        ],
    )
    .unwrap();
    for (tensor, representation) in [
        (&lazy, BatchMemberRepresentation::LazyAdjoint),
        (&diagonal, BatchMemberRepresentation::CompactDiagonal),
    ] {
        assert_eq!(
            StackedTensorMap::pack(&[&square, tensor]).err(),
            Some(Error::UnsupportedBatchMember {
                member: 1,
                representation
            })
        );
    }
}

#[test]
fn empty_batch_is_rejected() {
    let empty: [&TensorMap<fixtures::U1FusionRule, f64>; 0] = [];
    assert!(matches!(
        StackedTensorMap::pack(&empty),
        Err(Error::InvalidArgument(_))
    ));
}

/// Reordering, a subset (B' < B), the identity (B' = B), duplicates
/// (B' > B), and a selection of a selection.
const SELECTIONS: [&[usize]; 5] = [&[2, 0, 1], &[1], &[0, 1, 2], &[1, 1, 0, 2, 2, 1], &[2, 2]];

macro_rules! select_members {
    ($engine:ident, $label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let runtime = host_runtime();
        select_members!($engine, @dtype $label, &runtime, &a, f64);
        select_members!($engine, @dtype $label, &runtime, &a, Complex64);
    }};
    ($engine:ident, @dtype $label:expr, $runtime:expr, $a:expr, $d:ty) => {{
        let members = members!($engine, $runtime, $a, $d, 3);
        let stack = StackedTensorMap::pack(&members).unwrap();
        for selection in SELECTIONS {
            let label = format!("{} {} {:?}", $label, stringify!($d), selection);
            let selected = stack.select($engine, selection).unwrap();
            assert_eq!(selected.len(), selection.len(), "{label}");
            assert!(*selected.signature() == *stack.signature(), "{label}");
            for (j, &i) in selection.iter().enumerate() {
                let member = selected.member($engine, j).unwrap();
                assert!(member.structure_signature() == members[i].structure_signature());
                fixtures::assert_bit_exact(member.dense_data().unwrap(), members[i].dense_data().unwrap(), &label);
            }
            let again = selected.select($engine, &[selection.len() - 1, 0]).unwrap();
            fixtures::assert_bit_exact(
                again.member($engine, 0).unwrap().dense_data().unwrap(),
                members[selection[selection.len() - 1]].dense_data().unwrap(),
                &label,
            );
            fixtures::assert_bit_exact(
                again.member($engine, 1).unwrap().dense_data().unwrap(),
                members[selection[0]].dense_data().unwrap(),
                &label,
            );
        }
    }};
}

#[test]
fn select_holds_the_chosen_members_in_order() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    for_each_symmetry!(engine, select_members);
}

#[test]
fn select_rejects_out_of_range_and_empty_selections() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let q = fixtures::U1Irrep::new;
    let bond =
        GradedSpace::try_new(Arc::new(fixtures::U1FusionRule), [(q(0), 2), (q(1), 1)]).unwrap();
    let members = members!(engine, &runtime, &bond, f64, 3);
    let stack = StackedTensorMap::pack(&members).unwrap();
    assert_eq!(
        stack.select(engine, &[0, 3, 7]).err(),
        Some(Error::BatchMemberOutOfRange { member: 3, len: 3 })
    );
    assert!(matches!(
        stack.select(engine, &[]),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn select_of_a_blockless_structure_is_empty() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();
    let q = fixtures::U1Irrep::new;
    let charged = GradedSpace::try_new(Arc::new(fixtures::U1FusionRule), [(q(1), 2)]).unwrap();
    let neutral = GradedSpace::try_new(Arc::new(fixtures::U1FusionRule), [(q(0), 2)]).unwrap();
    let empty = TensorMap::<_, f64>::zeros(engine, &runtime, [&charged], [&neutral]).unwrap();
    assert!(empty.dense_data().unwrap().is_empty());
    let stack = StackedTensorMap::pack(&[&empty, &empty]).unwrap();
    let selected = stack.select(engine, &[1, 0, 1]).unwrap();
    assert_eq!(selected.len(), 3);
    assert!(selected
        .member(engine, 2)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}
