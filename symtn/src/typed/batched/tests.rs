use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use std::sync::Arc;

use symtn_core::{Placement, RuleIdentity, TensorStorage, U1FusionRule, U1Irrep};

use super::super::Runtime;
use super::super::{owned_repr, GradedSpace, TensorMap, TypedTensorBody};

#[test]
fn spectra_publication_failure_does_not_publish_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let x = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&leg], [&leg], 1).unwrap();
    let x_adjoint = &x.adjoint(engine).unwrap();
    let input = x.axpby(engine, 1.0, x_adjoint, 1.0).unwrap();
    let stack = super::StackedTensorMap::pack(&[input]).unwrap();
    let mut plan =
        super::EighFullPlan::new(engine, &stack, &[0], &[1], super::HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.routes.label_routes[0] = Some(usize::MAX);

    assert!(matches!(
        plan.execute(engine, &stack, &mut workspace),
        Err(super::BatchError::Operation(crate::typed::Error::InvalidArgument(reason)))
            if reason.contains("coupled sector has no route")
    ));
    assert!(workspace.take_output().is_none());
}

#[test]
fn compose_failure_after_the_output_is_taken_keeps_it_unobservable() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: an error inside the replay, after the output buffer is
    // taken (forced by an operand whose layout and payload are
    // consistently one entry short, which only a corrupted stack
    // carries), leaves no observable output; the buffer is kept and the
    // next call recovers.
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v, &v], [&v], 3).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v], [&v], 4).unwrap();
    let lhs = super::StackedTensorMap::pack(&[&a, &a]).unwrap();
    let rhs = super::StackedTensorMap::pack(&[&b, &b]).unwrap();
    let mut corrupt = super::StackedTensorMap::pack(&[&a, &a]).unwrap();
    corrupt.member_len -= 1;
    corrupt
        .storage
        .truncate(corrupt.member_len * corrupt.members);
    let plan = super::ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let expected = plan
        .execute(engine, &lhs, &rhs, &mut workspace)
        .unwrap()
        .storage
        .clone();
    let bytes = workspace.retained_bytes();

    // Only the route executor compares the payload with the plan's
    // operand structure; every earlier check passes.
    let Err(crate::error::Error::Operation(error)) = plan
        .execute(engine, &corrupt, &rhs, &mut workspace)
        .map(|_| ())
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
        plan.execute(engine, &lhs, &rhs, &mut workspace)
            .unwrap()
            .storage,
        expected
    );
    assert!(workspace.take_output().is_some());
}

#[test]
fn host_eigenvector_stack_equals_eager_bitwise_over_multi_tree_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: over sectors with several codomain trees, each member's `d` and
    // `v` are eager's bits.
    use super::EighFullPlan;
    use symtn_core::{SU2FusionRule, SU2Irrep};

    let runtime = Runtime::builder().build().unwrap();
    let j = SU2Irrep::from_twice_spin;
    let leg =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap();
    let members: Vec<_> = (0..3)
        .map(|seed| {
            let x = TensorMap::<_, f64>::rand_with_seed(
                engine,
                &runtime,
                [&leg, &leg],
                [&leg, &leg],
                seed,
            )
            .unwrap();
            let x_adjoint = &x.adjoint(engine).unwrap();
            x.axpby(engine, 1.0, x_adjoint, 1.0).unwrap()
        })
        .collect();
    let stack = super::StackedTensorMap::pack(&members).unwrap();
    let plan = EighFullPlan::new(
        engine,
        &stack,
        &[0, 1],
        &[2, 3],
        super::HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut workspace = plan.workspace().unwrap();
    let output = plan.execute(engine, &stack, &mut workspace).unwrap();
    let (stack_d, stack_v) = (output.d.storage.clone(), output.v.storage.clone());
    let d_len = stack_d.len() / members.len();
    let v_len = stack_v.len() / members.len();
    for (member, input) in members.iter().enumerate() {
        let crate::typed::Eigh { d, v } = input
            .eigh_full(engine, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT)
            .unwrap();
        assert!(stack_v[member * v_len..][..v_len] == *v.dense_data().unwrap());
        assert!(
            stack_d[member * d_len..][..d_len]
                == *d.materialize(engine).unwrap().dense_data().unwrap()
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_eigenvector_stack_equals_eager_bitwise_over_multi_tree_sectors() {
    let cpu = &mut crate::typed::CpuEngine::serial();
    let cuda =
        &mut crate::typed::CudaEngine::open(crate::typed::CudaDeviceId::from_ordinal(0)).unwrap();
    // What: over sectors with several codomain trees, the device `v` at
    // B = 1 is eager's device `v` bit for bit.
    use super::EighFullPlan;
    use symtn_core::{SU2FusionRule, SU2Irrep};

    let runtime = Runtime::builder().build().unwrap();
    let j = SU2Irrep::from_twice_spin;
    let leg =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap();
    let members: Vec<_> = (0..1)
        .map(|seed| {
            let x = TensorMap::<_, f64>::rand_with_seed(
                &mut *cpu,
                &runtime,
                [&leg, &leg],
                [&leg, &leg],
                seed,
            )
            .unwrap();
            let adjoint = x.adjoint(&mut *cpu).unwrap();
            x.axpby(&mut *cpu, 1.0, &adjoint, 1.0).unwrap()
        })
        .collect();
    let single = super::StackedTensorMap::pack(&members[..1])
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let plan = EighFullPlan::new(
        cuda,
        &single,
        &[0, 1],
        &[2, 3],
        super::HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut workspace = plan.workspace().unwrap();
    let v = plan
        .execute(cuda, &single, &mut workspace)
        .unwrap()
        .v
        .to_host(cuda)
        .unwrap()
        .member(cpu, 0)
        .unwrap();
    let crate::typed::Eigh { v: eager_v, .. } = members[0]
        .to_cuda(cuda)
        .unwrap()
        .eigh_full(cuda, &[0, 1], &[2, 3], super::HermitianTol::DEFAULT)
        .unwrap();
    assert!(v.dense_data().unwrap() == eager_v.to_host(cuda).unwrap().dense_data().unwrap());
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the same space and Runtime on different placements (Host,
    // two device ordinals) gives unequal signatures; equal placement
    // gives equal signatures and equal hashes.
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 3),
        ],
    )
    .unwrap();
    let host = TensorMap::<_, f64>::zeros(engine, &runtime, [&leg, &leg], [&leg]).unwrap();
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: replacing one field of an equal signature with another
    // signature's value is reported as exactly that field, and `==`
    // agrees with `first_mismatch`.
    use super::SignatureField;

    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let q = U1Irrep::new;
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 2), (q(1), 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 3), (q(1), 1)]).unwrap();
    let mut signature = |runtime: &Runtime, leg: &GradedSpace<U1FusionRule>| {
        TensorMap::<_, f64>::zeros(engine, runtime, [leg, leg], [leg])
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
    let engine = &mut crate::typed::CpuEngine::serial();
    // What: SU(3) and SU(4) trivial legs give the same hom space and
    // block structure content, so the rule identity alone separates them.
    use std::sync::Arc;
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let rule = |rank: usize| {
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(rank).unwrap()).unwrap())
    };
    // Why the rules outlive their signatures: a rule identity holds its group
    // weakly, so signatures of dropped rules equal nothing but their clones.
    let (first_su3, second_su3, su4_rule) = (rule(3), rule(3), rule(4));
    let mut signature = |rule: &Arc<LieGroupFusionRule>, trivial: Vec<i64>| {
        let leg = GradedSpace::try_new(Arc::clone(rule), [(trivial, 2)]).unwrap();
        TensorMap::<_, f64>::zeros(engine, &runtime, [&leg], [&leg])
            .unwrap()
            .structure_signature()
    };
    let su3 = signature(&first_su3, vec![0, 0]);
    let su4 = signature(&su4_rule, vec![0, 0, 0]);
    assert!(su3.homspace == su4.homspace);
    assert!(*su3.structure == *su4.structure);
    assert!(su3.placement == su4.placement && su3.runtime == su4.runtime);
    assert!(su3.rule != su4.rule);
    assert!(su3 != su4);
    assert!(su3 == signature(&second_su3, vec![0, 0]));
}
