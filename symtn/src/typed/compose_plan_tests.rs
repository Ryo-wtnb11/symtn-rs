//! `ComposePlan` tests that need private plan and stack fields (#1775): the
//! composition's route, the errors of corrupted stacks, and the CUDA zero
//! regions of the inactive destination blocks against a hand derivation
//! (#1980).

use std::sync::Arc;

use super::*;
use crate::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, U1FusionRule, U1Irrep, Z2Irrep,
};
#[cfg(feature = "cuda")]
use crate::sector::{SU2FusionRule, SU2Irrep};
use crate::typed::GradedSpace;
#[cfg(feature = "cuda")]
use std::collections::HashSet;

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let q = U1Irrep::new;
    let rule = Arc::new(U1FusionRule);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(q(-1), 2), (q(0), 1), (q(1), 3)]).unwrap(),
        GradedSpace::try_new(rule, [(q(0), 2), (q(1), 1)]).unwrap(),
    )
}

#[cfg(feature = "cuda")]
fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let j = SU2Irrep::from_twice_spin;
    let rule = Arc::new(SU2FusionRule);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(j(0), 2), (j(1), 2), (j(2), 1)]).unwrap(),
        GradedSpace::try_new(rule, [(j(0), 1), (j(2), 2)]).unwrap(),
    )
}

/// The composition's route: the unswapped direct core with every job
/// coefficient +1, never a candidate walk, `copyC` or `DynamicTree`.
fn assert_unit_unswapped_core<R>(what: &str, plan: &ComposePlan<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let resolution = &plan.0.resolution;
    assert!(
        resolution.copy_c().is_none() && !resolution.is_dynamic_tree(),
        "{what}"
    );
    let (core, swapped) = resolution.direct_core().unwrap();
    assert!(!swapped, "{what}");
    core.require_identity_direct_replay().unwrap();
}

#[test]
fn composition_resolves_to_the_unswapped_unit_core() {
    // What: on the fermionic `[v] <- [w']` · `[w'] <- [v]` geometry of
    // `signed_direct_can_leave_inactive_destination_blocks`, the composition
    // is the unit core while the contraction over the same legs is the
    // signed one: the kind, not the axes, decides the twist.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let lhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 1).unwrap();
    let rhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 2).unwrap();
    let plan = ComposePlan::new(
        &StackedTensorMap::pack(&[&lhs]).unwrap(),
        &StackedTensorMap::pack(&[&rhs]).unwrap(),
    )
    .unwrap();
    assert_unit_unswapped_core("U1", &plan);

    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
    let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
    let v =
        GradedSpace::try_new(Arc::clone(&rule), [(even(0), 2), (odd(1), 2), (odd(-1), 1)]).unwrap();
    let wd = GradedSpace::try_new(rule, [(even(0), 2), (odd(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let lhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v], [&wd], |_, _| 1.0).unwrap();
    let rhs = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&wd], [&v], |_, _| 1.0).unwrap();
    let left = StackedTensorMap::pack(&[&lhs]).unwrap();
    let right = StackedTensorMap::pack(&[&rhs]).unwrap();
    let plan = ComposePlan::new(&left, &right).unwrap();
    assert_unit_unswapped_core("fZ2xU1 dual leg", &plan);
    assert!(!plan
        .0
        .resolution
        .core_plan()
        .inactive_destination_regions()
        .is_empty());
    let spec = super::super::super::ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    let contract = ContractPlan::new(&left, &right, &spec).unwrap();
    let (core, swapped) = contract.resolution.direct_core().unwrap();
    assert!(!swapped);
    core.require_identity_signed_direct_replay().unwrap();
    assert!(core.require_identity_direct_replay().is_err());
}

#[test]
fn corrupted_stack_errors_are_typed() {
    // What: a stack whose payload disagrees with its structure (reachable
    // only through these private fields) is rejected with a typed error, and
    // a destination likewise.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (v, w) = u1_legs();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 6).unwrap();
    let lhs = StackedTensorMap::pack(&[&a, &a]).unwrap();
    let rhs = StackedTensorMap::pack(&[&b, &b]).unwrap();
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();
    let mut dst = workspace.take_output().unwrap();

    // Length and structure shrunk together: only the executor sees it.
    let mut consistent = StackedTensorMap::pack(&[&a, &a]).unwrap();
    consistent.member_len -= 1;
    consistent
        .storage
        .truncate(consistent.member_len * consistent.members);
    // Payload one entry short of its structure.
    let mut short = StackedTensorMap::pack(&[&a, &a]).unwrap();
    short.storage.pop();
    let mut short_dst = StackedTensorMap::pack(&[&a, &a]).unwrap();
    short_dst.space = dst.space.clone();
    short_dst.signature = dst.signature.clone();
    short_dst.member_len = dst.member_len;
    short_dst.storage = vec![1.0; dst.storage.len() - 1];

    let observed = [
        plan.execute(&consistent, &rhs, &mut workspace).map(|_| ()),
        plan.execute(&short, &rhs, &mut workspace).map(|_| ()),
        plan.execute_into(&short, &rhs, &mut dst, &mut workspace),
        plan.execute_into(&lhs, &rhs, &mut short_dst, &mut workspace),
    ]
    .map(Result::err);
    assert!(short_dst.storage.iter().all(|&value| value == 1.0));
    let payload =
        Error::InvalidArgument("stacked payload length differs from its structure".into());
    // `ComposePlan` runs `ContractPlan`'s per-call checks (#1775, stricter
    // than TensorKit): a payload whose length differs from its structure is
    // an `InvalidArgument` before any work. The consistent corruption reaches
    // the route executor, which reports the operand's length (two members of
    // 31) against the plan's (two of 32).
    assert_eq!(
        observed,
        [
            Some(Error::Operation(Box::new(
                OperationError::ElementCountMismatch {
                    expected: 64,
                    actual: 62
                }
            ))),
            Some(payload.clone()),
            Some(payload),
            Some(Error::InvalidArgument(
                "destination payload length differs from its structure".into()
            )),
        ]
    );
}

/// Every linear position of a strided region.
#[cfg(feature = "cuda")]
fn positions(dims: &[usize], strides: &[usize], offset: usize, out: &mut Vec<usize>) {
    let mut index = vec![0; dims.len()];
    if dims.contains(&0) {
        return;
    }
    loop {
        out.push(offset + index.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>());
        let Some(axis) = (0..dims.len()).find(|&axis| index[axis] + 1 < dims[axis]) else {
            return;
        };
        index[axis] += 1;
        index[..axis].fill(0);
    }
}

/// The payload positions, over a `members`-member stack, of the destination
/// blocks of `a · b` that no GEMM writes, derived from the public block API
/// alone: a block of the eager result whose coupled sector is not a coupled
/// sector of both operands receives no product. Each position appears once.
#[cfg(feature = "cuda")]
fn hand_inactive_positions<R, D>(
    a: &TensorMap<R, D>,
    b: &TensorMap<R, D>,
    members: usize,
) -> Vec<usize>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let coupled = |t: &TensorMap<R, D>, i: usize| {
        format!("{:?}", t.subblock_fusion_trees(i).unwrap().coupled())
    };
    let sectors = |t: &TensorMap<R, D>| {
        (0..t.subblock_count())
            .map(|i| coupled(t, i))
            .collect::<HashSet<_>>()
    };
    let active: HashSet<_> = sectors(a).intersection(&sectors(b)).cloned().collect();
    let c = a.compose(b).unwrap();
    let member_len = c.dense_data().unwrap().len();
    let mut out = Vec::new();
    for i in (0..c.subblock_count()).filter(|&i| !active.contains(&coupled(&c, i))) {
        let block = c.subblock(i).unwrap();
        for member in 0..members {
            positions(
                block.shape(),
                block.strides(),
                block.offset() + member * member_len,
                &mut out,
            );
        }
    }
    out.sort_unstable();
    assert!(!out.is_empty(), "the fixture has inactive blocks");
    out
}

/// The zero regions `regions` cover exactly the hand-derived inactive
/// positions, each once.
#[cfg(feature = "cuda")]
fn assert_regions_cover<R, D>(
    what: &str,
    regions: &[symtn_dense::CudaRegion],
    a: &TensorMap<R, D>,
    b: &TensorMap<R, D>,
    members: usize,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    let mut covered = Vec::new();
    for region in regions {
        eprintln!(
            "{what} B={members}: dims {:?} strides {:?} offset {}",
            region.dims(),
            region.strides(),
            region.offset()
        );
        positions(
            region.dims(),
            region.strides(),
            region.offset(),
            &mut covered,
        );
    }
    covered.sort_unstable();
    assert_eq!(
        covered,
        hand_inactive_positions(a, b, members),
        "{what} B={members}"
    );
}

/// The zero regions the CUDA member stage builds for a composition's core
/// (`MemberCudaStage::prepare` → `CudaMemberZeroRegions::prepare` over the
/// core plan's inactive blocks), without a device: the resolution does not
/// depend on the placement.
#[cfg(feature = "cuda")]
fn regions_match_hand_derivation<R>(
    runtime: &Runtime,
    what: &str,
    (v, w): (GradedSpace<R>, GradedSpace<R>),
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let a = TensorMap::<_, f64>::rand_with_seed(runtime, [&v, &v], [&w], 11).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(runtime, [&w], [&v], 12).unwrap();
    let plan = ComposePlan::new(
        &StackedTensorMap::pack(&[&a]).unwrap(),
        &StackedTensorMap::pack(&[&b]).unwrap(),
    )
    .unwrap();
    let resolution = &plan.0.resolution;
    assert!(resolution.admit_cuda_members().unwrap() > 0, "{what}");
    for members in [1, 2, 17] {
        let regions = symtn_operations::cuda_transform::CudaMemberZeroRegions::prepare(
            resolution.core_plan().inactive_destination_regions(),
            plan.0.member_len,
            members,
        )
        .unwrap();
        assert_regions_cover(what, regions.regions(), &a, &b, members);
    }
}

/// #1980, #1775: the regions come from the one member stage, which also
/// validates each within the stack.
#[cfg(feature = "cuda")]
#[test]
fn cuda_zero_regions_match_the_hand_derivation() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    regions_match_hand_derivation(&runtime, "U1", u1_legs());
    regions_match_hand_derivation(&runtime, "SU2", su2_legs());
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
    let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
    let fz2u1 = (
        GradedSpace::try_new(Arc::clone(&rule), [(even(0), 2), (odd(1), 1), (odd(-1), 2)]).unwrap(),
        GradedSpace::try_new(rule, [(even(0), 1), (odd(1), 2)]).unwrap(),
    );
    regions_match_hand_derivation(&runtime, "fZ2xU1", fz2u1);
}
