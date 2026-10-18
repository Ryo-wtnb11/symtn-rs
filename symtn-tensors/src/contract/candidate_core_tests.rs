//! Route pins for the zero-copy TensorKit candidates (#1468): a contraction
//! whose sources are the identity only after the paired-axis sort or the
//! operand swap, including through a lazy adjoint, resolves to the direct
//! Core GEMM and never projects or transforms an operand. Independent value
//! oracles live in `symtn/tests/contract_candidate_core.rs`.

use crate::probe::{self, Event};
use std::sync::Arc;

use symtn_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols,
    ProductFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, SectorId, SectorLeg,
    U1FusionRule, U1Irrep,
};
use symtn_operations::fusion_replay::{Rank2Gemm, Rank2GemmBatchJob};
use symtn_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use symtn_operations::{OutputAxisOrder, TensorContractSpec};

use crate::contract::fusion::FusionContractOrientation;
use crate::{BoundDynamicFusionMapSpace, FusionOperand, RuleIdentity};

type Context = crate::TensorContractFusionExecutionContext<f64, RuleIdentity>;
type FermionU1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

#[derive(Default)]
struct CountBatchCalls {
    calls: usize,
    jobs: usize,
}

impl Rank2Gemm<f64> for CountBatchCalls {
    fn matmul_rank2(
        &mut self,
        _: &mut [f64],
        _: &[f64],
        _: &[f64],
        _: usize,
        _: usize,
        _: usize,
        _: f64,
        _: f64,
    ) -> Result<(), crate::OperationError> {
        panic!("stacked direct replay must submit one batch")
    }

    fn matmul_rank2_batch(
        &mut self,
        _: &mut [f64],
        _: &[f64],
        _: &[f64],
        jobs: &[Rank2GemmBatchJob],
        _: &[usize],
        alpha: f64,
        beta: f64,
    ) -> Result<(), crate::OperationError> {
        assert_eq!((alpha, beta), (1.0, 0.0));
        self.calls += 1;
        self.jobs += jobs.len();
        Ok(())
    }
}

fn assert_one_batch_submission(
    route: &crate::StorageContractResolution<f64>,
    dst_len: usize,
    lhs_len: usize,
    rhs_len: usize,
    swapped: bool,
) -> usize {
    let mut jobs_per_member = 0;
    for members in [1, 2, 17] {
        let (core, actual_swapped) = route.direct_core().unwrap();
        core.require_identity_direct_replay().unwrap();
        let replay = StackedDirectReplay::new_signed(Arc::clone(core), members).unwrap();
        assert_eq!(actual_swapped, swapped);
        let mut dst = vec![f64::NAN; dst_len * members];
        let lhs = vec![1.0; lhs_len * members];
        let rhs = vec![1.0; rhs_len * members];
        let (left, right) = if swapped { (&rhs, &lhs) } else { (&lhs, &rhs) };
        let (left_len, right_len) = if swapped {
            (rhs_len, lhs_len)
        } else {
            (lhs_len, rhs_len)
        };
        let left = StackedStorageView::new::<f64>(left, left_len, members, left_len).unwrap();
        let right = StackedStorageView::new::<f64>(right, right_len, members, right_len).unwrap();
        let mut dst =
            StackedStorageViewMut::new::<f64>(&mut dst, dst_len, members, dst_len).unwrap();
        let mut gemm = CountBatchCalls::default();
        replay
            .execute_host(
                &mut crate::StridedHostKernelAdapter::default(),
                &mut gemm,
                &mut dst,
                &left,
                &right,
                true,
            )
            .unwrap();
        if members == 1 {
            jobs_per_member = gemm.jobs;
            assert!(jobs_per_member > 0);
        }
        assert_eq!((gemm.calls, gemm.jobs), (1, jobs_per_member * members));
    }
    jobs_per_member
}

fn square<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    codomain: &SectorLeg,
    domain: &SectorLeg,
) -> BoundDynamicFusionMapSpace<R> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([codomain.clone(), codomain.clone()]),
            FusionProductSpace::new([domain.clone(), domain.clone()]),
        ),
    )
    .unwrap()
}

/// Non-self-dual: `-1` and `1` carry different degeneracies.
fn u1_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 1),
            (U1Irrep::new(1).sector_id(), 3),
        ],
        false,
    )
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [
            (SU2Irrep::from_twice_spin(0).sector_id(), 2),
            (SU2Irrep::from_twice_spin(1).sector_id(), 1),
            (SU2Irrep::from_twice_spin(2).sector_id(), 1),
        ],
        false,
    )
}

fn fermion_u1_leg(rule: &FermionU1, dual: bool) -> SectorLeg {
    let (even, odd) = (SectorId::new(0), SectorId::new(1));
    let charge = |q: i32| U1Irrep::new(if dual { -q } else { q }).sector_id();
    SectorLeg::new(
        [
            (rule.encode_component_ids(even, charge(0)), 2),
            (rule.encode_component_ids(odd, charge(1)), 2),
            (rule.encode_component_ids(odd, charge(-1)), 1),
            (rule.encode_component_ids(even, charge(1)), 1),
        ],
        dual,
    )
}

fn data(len: usize, salt: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 37 + salt * 11) % 17) as f64 / 8.0 - 1.0)
        .collect()
}

/// `(name, lhs lazy, rhs lazy, lhs axes, rhs axes, output, orientation)`:
/// the probes C1, C2, L3, L5, L7 plus the literal controls C0, L4.
type Probe = (
    &'static str,
    bool,
    bool,
    &'static [usize],
    &'static [usize],
    &'static [usize],
    FusionContractOrientation,
);

const PROBES: [Probe; 7] = [
    ("C0", false, false, &[2, 3], &[0, 1], &[0, 1, 2, 3], LHS_RHS),
    ("C1", false, false, &[3, 2], &[1, 0], &[0, 1, 2, 3], LHS_RHS),
    ("C2", false, false, &[0, 1], &[2, 3], &[2, 3, 0, 1], RHS_LHS),
    ("L4", true, false, &[2, 3], &[0, 1], &[0, 1, 2, 3], LHS_RHS),
    ("L3", true, false, &[3, 2], &[1, 0], &[0, 1, 2, 3], LHS_RHS),
    ("L5", false, true, &[0, 1], &[2, 3], &[2, 3, 0, 1], RHS_LHS),
    ("L7", false, true, &[3, 2], &[1, 0], &[0, 1, 2, 3], LHS_RHS),
];
const LHS_RHS: FusionContractOrientation = FusionContractOrientation::LhsRhs;
const RHS_LHS: FusionContractOrientation = FusionContractOrientation::RhsLhs;

fn operand(space: &BoundDynamicFusionMapSpace<impl Sized>, lazy: bool) -> FusionOperand<'_> {
    if lazy {
        FusionOperand::adjoint(space.space())
    } else {
        FusionOperand::direct(space.space())
    }
}

/// Every probe on `V⊗V ← V⊗V` operands resolves to Core in its TensorKit
/// orientation, with no logical projection or adjoint view of a lazy
/// operand, on the Host and on the device compile route.
fn assert_probes_resolve_to_core<R>(provider: &Arc<R>, leg: &SectorLeg)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + crate::TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let space = square(provider, leg, leg);
    let len = space.space().required_len().unwrap();
    let (lhs_data, rhs_data) = (data(len, 1), data(len, 2));
    let mut context = Context::default();
    for (name, lhs_lazy, rhs_lazy, lhs_axes, rhs_axes, output, orientation) in PROBES {
        let axes = TensorContractSpec::new_with_conjugation(
            lhs_axes,
            rhs_axes,
            OutputAxisOrder::from_axes(output),
            lhs_lazy,
            rhs_lazy,
        );
        let mut dst = vec![0.0; len];
        let adjoint_view_build_count_rec = probe::record();
        let fusion_operand_projection_prepares_rec = probe::record();
        if lhs_lazy || rhs_lazy {
            context
                .tensorcontract_prelowered_into(
                    &space,
                    &mut dst,
                    operand(&space, lhs_lazy),
                    &lhs_data,
                    operand(&space, rhs_lazy),
                    &rhs_data,
                    axes,
                    1.0,
                    crate::ContractDestinationInit::Axpby(0.0),
                )
                .unwrap();
        } else {
            context
                .tensorcontract_prelowered_into(
                    &space,
                    &mut dst,
                    crate::FusionOperand::direct(space.space()),
                    &lhs_data,
                    crate::FusionOperand::direct(space.space()),
                    &rhs_data,
                    axes,
                    1.0,
                    crate::ContractDestinationInit::Axpby(0.0),
                )
                .unwrap();
        }
        // What: the selected zero-copy candidate runs the direct Core GEMM.
        assert!(context.last_resolution_is_core(), "{name}");
        let swapped = (orientation == RHS_LHS).then_some(RHS_LHS);
        assert_eq!(context.last_resolution_orientation(), swapped, "{name}");
        // What: no logical key projection or adjoint view of a lazy operand.
        assert_eq!(
            fusion_operand_projection_prepares_rec.get(Event::FusionOperandProjectionPrepare),
            0,
            "{name}"
        );
        assert_eq!(
            adjoint_view_build_count_rec.get(Event::AdjointViewBuild),
            0,
            "{name}"
        );
        assert!(dst.iter().any(|&x| x != 0.0), "{name}: empty fixture");

        // What: the device compile route takes the same Core candidate.
        let resolution =
            crate::try_compile_storage_contract_core_route::<crate::DirectCoreExecutor, _>(
                &space,
                operand(&space, lhs_lazy),
                operand(&space, rhs_lazy),
                axes,
            )
            .unwrap()
            .hit()
            .unwrap_or_else(|| panic!("{name}: device route declined Core"));
        assert!(!resolution.is_dynamic_tree(), "{name}");
        assert_eq!(
            resolution.is_swapped_core(),
            orientation == RHS_LHS,
            "{name}"
        );
        if name == "C0" {
            resolution
                .direct_core()
                .unwrap()
                .0
                .require_identity_direct_replay()
                .unwrap();
        }
        if provider.braiding_style() == symtn_core::BraidingStyleKind::Bosonic
            && matches!(name, "C0" | "C2")
        {
            assert_one_batch_submission(&resolution, len, len, len, orientation == RHS_LHS);
        }
    }
}

#[test]
fn zero_copy_candidates_resolve_to_core_on_non_self_dual_u1() {
    assert_probes_resolve_to_core(&Arc::new(U1FusionRule), &u1_leg());
}

#[test]
fn zero_copy_candidates_resolve_to_core_on_su2() {
    assert_probes_resolve_to_core(&Arc::new(SU2FusionRule), &su2_leg());
}

#[test]
fn asymmetric_owned_candidate_reaches_swapped_storage_core() {
    let provider = Arc::new(U1FusionRule);
    let (v, w, x) = (
        u1_leg(),
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 1),
                (U1Irrep::new(1).sector_id(), 2),
            ],
            false,
        ),
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 2),
                (U1Irrep::new(1).sector_id(), 3),
            ],
            false,
        ),
    );
    let lhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([v.clone()]),
            FusionProductSpace::new([w]),
        ),
    )
    .unwrap();
    let rhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        provider,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([x.clone(), x]),
            FusionProductSpace::new([v]),
        ),
    )
    .unwrap();
    let order = OutputAxisOrder::from_axes(&[1, 2, 0]);
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
        &lhs,
        &rhs,
        &[0],
        &[2],
        order,
        2,
    )
    .unwrap();
    let route = Context::default()
        .compile_storage_contract_resolution(
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&[0], &[2], order),
        )
        .unwrap();
    assert!(route.is_swapped_core());
    let jobs = assert_one_batch_submission(
        &route,
        dst.space().required_len().unwrap(),
        lhs.space().required_len().unwrap(),
        rhs.space().required_len().unwrap(),
        true,
    );
    assert_eq!(jobs, 2);
    assert_ne!(
        lhs.space().required_len().unwrap(),
        rhs.space().required_len().unwrap()
    );
}

#[test]
fn zero_copy_candidates_resolve_to_core_on_fermion_u1() {
    let rule = FermionParityFusionRule.product(U1FusionRule);
    let leg = fermion_u1_leg(&rule, false);
    assert_probes_resolve_to_core(&Arc::new(rule), &leg);
}

/// `A: V⊗V ← V*⊗V*` against `B: V*⊗V* ← V⊗V` sorted into `mul!` form puts
/// dual legs on B's contracted codomain. TensorKit `blas_contract!` copies
/// one operand to twist it; θ here is the parity of B's coupled sector, so
/// the Host core carries it as per-job GEMM alpha and copies nothing (#1858).
/// Values against TensorKit's steps: `symtn/tests/typed_contract_host_oracle.rs`.
#[test]
fn a_uniform_fermionic_twist_rides_the_sorted_candidate_core_as_gemm_alpha() {
    let rule = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let (v, v_dual) = (fermion_u1_leg(&rule, false), fermion_u1_leg(&rule, true));
    let lhs = square(&rule, &v, &v_dual);
    let rhs = square(&rule, &v_dual, &v);
    let dst = square(&rule, &v, &v);
    let axes = TensorContractSpec::new(&[3, 2], &[1, 0], OutputAxisOrder::identity());
    let mut out = vec![0.0; dst.space().required_len().unwrap()];
    let mut context = Context::default();
    context
        .tensorcontract_prelowered_into(
            &dst,
            &mut out,
            crate::FusionOperand::direct(lhs.space()),
            &data(lhs.space().required_len().unwrap(), 3),
            crate::FusionOperand::direct(rhs.space()),
            &data(rhs.space().required_len().unwrap(), 4),
            axes,
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();
    assert!(context.last_resolution_is_core());
}
