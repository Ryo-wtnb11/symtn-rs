//! Public network conformance for representative multiplicity-free providers.
//!
//! The direct typed operations are the oracle: this file checks that the
//! cached `Network::contract` and an explicitly planned `Network` preserve their provider and
//! payload, including a static intra-operand trace.

use std::sync::Arc;
use symtn::typed::ContractSpec;
use symtn::typed::Engine;

use symtn::sector::{
    product_sector, CU1FusionRule, CU1Irrep, FermionParityFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
};
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::{plan_cache_stats, GreedyDenseOptimizer, Network, NetworkExecutionWorkspace};

use crate::network_support;
use network_support::{net, op};

use crate::numerics;

/// The planned and cached network paths may group the contraction differently from the
/// direct typed call, so payloads agree within the workspace tolerance rule
/// with `terms` bounded by the contracted space's weighted dimension.
fn assert_same<R>(
    actual: &TensorMap<R, f64>,
    expected: &TensorMap<R, f64>,
    provider: &R,
    terms: usize,
) where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    assert!(std::ptr::eq(actual.provider(), provider));
    assert!(std::ptr::eq(expected.provider(), provider));
    assert!(actual
        .codomain()
        .iter()
        .all(|leg| std::ptr::eq(leg.provider(), provider)));
    assert!(actual
        .domain()
        .iter()
        .all(|leg| std::ptr::eq(leg.provider(), provider)));
    assert_eq!(actual.codomain(), expected.codomain());
    assert_eq!(actual.domain(), expected.domain());
    numerics::assert_slices_close(
        "payload",
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        terms,
    );
}

fn ordinary_network_and_workspace_reuse<R>(
    engine: &mut dyn Engine,
    runtime: &Runtime,
    space: &GradedSpace<R>,
    seed: u64,
) where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let lhs = TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space], seed).unwrap();
    let rhs =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space], seed + 1).unwrap();
    let terms = space.dim().unwrap().ceil() as usize;
    let expected = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .permute(engine, &[1], &[0])
        .unwrap();
    let network = Network::new(
        vec![vec!["i", "j"], vec!["j", "k"]],
        vec![false, false],
        vec![1, 1],
        vec!["k", "i"],
        Some(1),
    )
    .unwrap();
    let tensors = [&lhs, &rhs];
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    let first = planned.execute(engine, &tensors, &mut workspace).unwrap();
    let second = planned.execute(engine, &tensors, &mut workspace).unwrap();
    assert_same(&first, &expected, space.provider(), terms);
    assert_same(&second, &expected, space.provider(), terms);

    // This is workspace reuse only.  The public contract intentionally makes
    // no promise that output payload allocations themselves are reused.
    let cold = plan_cache_stats(runtime);
    let cached_first = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
        .contract(engine, &[&lhs, &rhs])
        .unwrap();
    let cached_second = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
        .contract(engine, &[&lhs, &rhs])
        .unwrap();
    assert_same(&cached_first, &expected, space.provider(), terms);
    assert_same(&cached_second, &expected, space.provider(), terms);
    assert!(plan_cache_stats(runtime).workspace_reuses > cold.workspace_reuses);
}
#[test]
fn multiplicity_free_public_network_path_matches_typed_oracles() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();

    let z2 = GradedSpace::try_new(
        Arc::new(Z2FusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(engine, &runtime, &z2, 1002);

    let z3_provider = Arc::new(ZNFusionRule::new(3).unwrap());
    let z3 = GradedSpace::try_new(
        Arc::clone(&z3_provider),
        [
            (z3_provider.irrep(0), 1),
            (z3_provider.irrep(1), 2),
            (z3_provider.irrep(2), 1),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(engine, &runtime, &z3, 1010);

    // A charged CU(1) leg exercises the nontrivial pseudo-scalar provider,
    // rather than a vacuum-only dense block.
    let cu1 = GradedSpace::try_new(
        Arc::new(CU1FusionRule),
        [
            (CU1Irrep::VACUUM, 1),
            (CU1Irrep::PSEUDOSCALAR, 2),
            (CU1Irrep::from_twice_charge(1), 1),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(engine, &runtime, &cu1, 1004);

    let product_rule = Arc::new(
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let product = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(engine, &runtime, &product, 1006);
    // The odd fZ2 sector contributes with the supertrace sign even after the
    // U(1) factor is attached: (2 + 3) - 7 = -2.  This is intentionally a
    // hand oracle, not another call to `trace_pairs`.
    let product_diagonal =
        TensorMap::from_subblock_fn(engine, &runtime, [&product], [&product], |trees, i| {
            if i[0] != i[1] {
                0.0
            } else if *trees.coupled() == product_sector(Z2Irrep::EVEN, U1Irrep::new(0)) {
                2.0 + i[0] as f64
            } else {
                7.0
            }
        })
        .unwrap();
    assert_eq!(
        product_diagonal
            .trace_pairs(engine, &[(0, 1)])
            .unwrap()
            .scalar(engine)
            .unwrap(),
        -2.0
    );

    let nested_rule = Arc::new(
        symtn_core::ProductFusionRule::<_, _>::try_new(
            symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
                .unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    );
    let nested = GradedSpace::try_new(
        nested_rule,
        [
            (
                product_sector(
                    product_sector(Z2Irrep::EVEN, U1Irrep::new(0)),
                    SU2Irrep::from_twice_spin(0),
                ),
                2,
            ),
            (
                product_sector(
                    product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
                    SU2Irrep::from_twice_spin(1),
                ),
                1,
            ),
        ],
    )
    .unwrap();
    ordinary_network_and_workspace_reuse(engine, &runtime, &nested, 1008);
}
