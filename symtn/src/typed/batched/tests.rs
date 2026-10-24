use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use std::sync::Arc;

use symtn_core::{Placement, RuleIdentity, TensorStorage, U1FusionRule, U1Irrep};

use super::super::Runtime;
use super::super::{owned_repr, GradedSpace, TensorMap, TypedTensorBody};

use super::super::probe::{self, RoutePolicy};

/// Forces every eigh route onto the per-tree copies, so the branch the
/// fixtures' aligned routes never take is exercised.
fn eigh_routes(treewise: bool) -> RoutePolicy {
    RoutePolicy {
        eigh_treewise: treewise,
        ..RoutePolicy::DEFAULT
    }
}

#[test]
fn spectra_publication_failure_does_not_publish_factors() {
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let x = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 1).unwrap();
    let input = x.axpby(1.0, &x.adjoint().unwrap(), 1.0).unwrap();
    let stack = super::StackedTensorMap::pack(&[input]).unwrap();
    let mut plan =
        super::EighFullPlan::new(&stack, &[0], &[1], super::HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.routes.label_routes[0] = Some(usize::MAX);

    assert!(plan.execute(&stack, &mut workspace).is_err());
    assert!(workspace.take_output().is_none());
}

#[test]
fn compose_failure_after_the_output_is_taken_keeps_it_unobservable() {
    // What: an error inside the replay, after the output buffer is
    // taken (forced by an operand whose layout and payload are
    // consistently one entry short, which only a corrupted stack
    // carries), leaves no observable output; the buffer is kept and the
    // next call recovers.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&v], 3).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 4).unwrap();
    let lhs = super::StackedTensorMap::pack(&[&a, &a]).unwrap();
    let rhs = super::StackedTensorMap::pack(&[&b, &b]).unwrap();
    let mut corrupt = super::StackedTensorMap::pack(&[&a, &a]).unwrap();
    corrupt.member_len -= 1;
    corrupt
        .storage
        .truncate(corrupt.member_len * corrupt.members);
    let plan = super::ComposePlan::new(&lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let expected = plan
        .execute(&lhs, &rhs, &mut workspace)
        .unwrap()
        .storage
        .clone();
    let bytes = workspace.retained_bytes();

    // Only the route executor compares the payload with the plan's
    // operand structure; every earlier check passes.
    let Err(crate::error::Error::Operation(error)) =
        plan.execute(&corrupt, &rhs, &mut workspace).map(|_| ())
    else {
        panic!("the corrupted stack must fail as an operation error");
    };
    assert!(
        matches!(
            *error,
            symtn_tensors::OperationError::ElementCountMismatch { .. }
        ),
        "{error:?}"
    );
    assert!(
        workspace.output.spare.is_some(),
        "the error is after the take"
    );
    assert!(workspace.take_output().is_none());
    assert_eq!(workspace.retained_bytes(), bytes);
    assert_eq!(
        plan.execute(&lhs, &rhs, &mut workspace).unwrap().storage,
        expected
    );
    assert!(workspace.take_output().is_some());
}

#[test]
fn host_tree_wise_eigenvector_copies_equal_the_aligned_ones_and_eager_bitwise() {
    // What: on Host the per-tree placement (one row slice per column
    // and codomain tree) writes exactly the bits of the aligned one (one
    // copy per sector), and both are eager's `d` and `v` per member.
    use super::EighFullPlan;
    use symtn_core::{SU2FusionRule, SU2Irrep};

    let runtime = Runtime::builder().build().unwrap();
    let j = SU2Irrep::from_twice_spin;
    let leg =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap();
    let members: Vec<_> = (0..3)
        .map(|seed| {
            let x = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], seed)
                .unwrap();
            x.axpby(1.0, &x.adjoint().unwrap(), 1.0).unwrap()
        })
        .collect();
    let stack = super::StackedTensorMap::pack(&members).unwrap();
    let run = |treewise: bool| {
        let plan = probe::with_routes(eigh_routes(treewise), || {
            EighFullPlan::new(&stack, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT).unwrap()
        });
        let mut workspace = plan.workspace().unwrap();
        let copies: usize = plan.routes.copies.iter().map(Vec::len).sum();
        let routes = plan.routes.copies.len();
        let output = plan.execute(&stack, &mut workspace).unwrap();
        (
            copies,
            routes,
            output.d.storage.clone(),
            output.v.storage.clone(),
        )
    };
    let (aligned_copies, routes, aligned_d, aligned_v) = run(false);
    let (tree_copies, _, tree_d, tree_v) = run(true);
    assert_eq!(aligned_copies, routes, "every fixture route is aligned");
    assert!(tree_copies > routes, "several codomain trees per sector");
    assert!(tree_v == aligned_v && tree_d == aligned_d);
    let d_len = aligned_d.len() / members.len();
    let v_len = aligned_v.len() / members.len();
    for (member, input) in members.iter().enumerate() {
        let crate::typed::Eigh { d, v } = input
            .eigh_full(&[0, 1], &[2, 3], super::HermitianTol::DEFAULT)
            .unwrap();
        assert!(tree_v[member * v_len..][..v_len] == *v.dense_data().unwrap());
        assert!(
            tree_d[member * d_len..][..d_len] == *d.materialize().unwrap().dense_data().unwrap()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn tree_wise_eigenvector_copies_equal_the_aligned_ones_bitwise() {
    // What: the per-tree placement (one copy per codomain tree) writes
    // exactly the bits of the aligned placement (one copy per sector),
    // and both are eager's device `v` at B = 1.
    use super::EighFullPlan;
    use symtn_core::{SU2FusionRule, SU2Irrep};

    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let j = SU2Irrep::from_twice_spin;
    let leg =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap();
    let members: Vec<_> = (0..3)
        .map(|seed| {
            let x = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], seed)
                .unwrap();
            x.axpby(1.0, &x.adjoint().unwrap(), 1.0).unwrap()
        })
        .collect();
    let stack = super::StackedTensorMap::pack(&members)
        .unwrap()
        .to_cuda()
        .unwrap();
    let run = |treewise: bool| {
        let plan = probe::with_routes(eigh_routes(treewise), || {
            EighFullPlan::new(&stack, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT).unwrap()
        });
        let mut workspace = plan.workspace().unwrap();
        let copies: usize = plan.routes.copies.iter().map(Vec::len).sum();
        let routes = plan.routes.copies.len();
        let output = plan.execute(&stack, &mut workspace).unwrap();
        (
            copies,
            routes,
            output.v.to_host().unwrap(),
            output.d.to_host().unwrap(),
        )
    };
    let (aligned_copies, routes, aligned_v, aligned_d) = run(false);
    let (tree_copies, _, tree_v, tree_d) = run(true);
    assert_eq!(aligned_copies, routes, "every fixture route is aligned");
    assert!(tree_copies > routes, "several codomain trees per sector");
    assert!(tree_v.storage == aligned_v.storage);
    assert!(tree_d.storage == aligned_d.storage);
    let single = super::StackedTensorMap::pack(&members[..1])
        .unwrap()
        .to_cuda()
        .unwrap();
    let plan = probe::with_routes(eigh_routes(true), || {
        EighFullPlan::new(&single, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT).unwrap()
    });
    let mut workspace = plan.workspace().unwrap();
    let v = plan
        .execute(&single, &mut workspace)
        .unwrap()
        .v
        .to_host()
        .unwrap()
        .member(0)
        .unwrap();
    let crate::typed::Eigh { v: eager_v, .. } = members[0]
        .to_cuda()
        .unwrap()
        .eigh_full(&[0, 1], &[2, 3], super::HermitianTol::DEFAULT)
        .unwrap();
    assert!(v.dense_data().unwrap() == eager_v.to_host().unwrap().dense_data().unwrap());
}

struct PlacedStorage {
    len: usize,
    placement: Placement,
}

impl TensorStorage<f64> for PlacedStorage {
    fn len(&self) -> usize {
        self.len
    }

    fn placement(&self) -> Placement {
        self.placement
    }
}

fn hash_of<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn placement_alone_separates_signatures() {
    // What: the same space and Runtime on different placements (Host,
    // two device ordinals) gives unequal signatures; equal placement
    // gives equal signatures and equal hashes.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap();
    let host = TensorMap::<_, f64>::zeros(&runtime, [&leg, &leg], [&leg]).unwrap();
    let placed = |placement| TensorMap::<_, f64, PlacedStorage> {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            host.logical_space().clone(),
            PlacedStorage {
                len: host.dense_data().unwrap().len(),
                placement,
            },
        )),
    };
    let host_signature = host.structure_signature();
    let cuda0 = placed(Placement::Cuda(0)).structure_signature();
    let cuda1 = placed(Placement::Cuda(1)).structure_signature();
    let host_again = placed(Placement::Host).structure_signature();

    assert!(host_signature != cuda0);
    assert!(cuda0 != cuda1);
    assert!(host_signature == host_again);
    assert_eq!(hash_of(&host_signature), hash_of(&host_again));
    assert!(cuda0 == placed(Placement::Cuda(0)).structure_signature());
}

#[test]
fn first_mismatch_names_each_field_alone() {
    // What: replacing one field of an equal signature with another
    // signature's value is reported as exactly that field, and `==`
    // agrees with `first_mismatch`.
    use super::SignatureField;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let q = U1Irrep::new;
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 2), (q(1), 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 3), (q(1), 1)]).unwrap();
    let signature = |runtime: &Runtime, leg: &GradedSpace<U1FusionRule>| {
        TensorMap::<_, f64>::zeros(runtime, [leg, leg], [leg])
            .unwrap()
            .structure_signature()
    };
    let base = signature(&runtime, &leg);
    let other = signature(&other_runtime, &wide);
    assert_eq!(base.first_mismatch(&base.clone()), None);

    let mut cases = Vec::new();
    let mut changed = base.clone();
    changed.placement = Placement::Cuda(0);
    cases.push((changed, SignatureField::Placement));
    let mut changed = base.clone();
    changed.runtime = other.runtime.clone();
    cases.push((changed, SignatureField::Runtime));
    let mut changed = base.clone();
    changed.rule = RuleIdentity::from_canonical_bytes::<u8>(1, Arc::from([]));
    cases.push((changed, SignatureField::Rule));
    let mut changed = base.clone();
    changed.homspace = other.homspace.clone();
    cases.push((changed, SignatureField::HomSpace));
    let mut changed = base.clone();
    changed.structure = other.structure.clone();
    cases.push((changed, SignatureField::BlockStructure));
    for (changed, field) in cases {
        assert_eq!(base.first_mismatch(&changed), Some(field));
        assert!(base != changed, "{field:?}");
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_rule_instance_alone_separates_signatures() {
    // What: SU(3) and SU(4) trivial legs give the same hom space and
    // block structure content, so the rule identity alone separates them.
    use std::sync::Arc;
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let signature = |rank: usize, trivial: Vec<i64>| {
        let rule = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(rank).unwrap()).unwrap(),
        );
        let leg = GradedSpace::try_new(rule, [(trivial, 2)]).unwrap();
        TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg])
            .unwrap()
            .structure_signature()
    };
    let su3 = signature(3, vec![0, 0]);
    let su4 = signature(4, vec![0, 0, 0]);
    assert!(su3.homspace == su4.homspace);
    assert!(*su3.structure == *su4.structure);
    assert!(su3.placement == su4.placement && su3.runtime == su4.runtime);
    assert!(su3.rule != su4.rule);
    assert!(su3 != su4);
    assert!(su3 == signature(3, vec![0, 0]));
}
