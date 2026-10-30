//! Typed Host network contraction through `Network::contract` (#2022):
//! provider and dtype matrix, high-rank and permuted outputs, `conj`
//! operands, rejections, and the typed workspace pools of the plan cache.

use std::sync::Arc;
use symtn::typed::ContractSpec;
use symtn::typed::Engine;

use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex64, Error, TensorScalar};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::plan_cache_stats;

use crate::network_support;
use network_support::{conj, net, op};

use crate::braiding_probe;

use crate::numerics;

fn u1_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

fn su2_space() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

fn assert_close(lhs: &[f64], rhs: &[f64], tol: f64) {
    assert_eq!(lhs.len(), rhs.len(), "data lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs())),
            "element {index} differs: {a} vs {b}"
        );
    }
}

fn assert_pair<R, D>(engine: &mut dyn Engine, a: &TensorMap<R, D>, b: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
    D: TensorScalar + Send + Sync + numerics::Numeric + 'static,
{
    // The cached plan may group the contraction differently from the direct call,
    // so the payloads agree under the tolerance rule; the contracted length is
    // bounded by the weighted dimension of the shared leg.
    let terms = a.domain()[0].dim().unwrap().ceil() as usize;
    let actual = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[a, b])
        .unwrap();
    let expected = a
        .contract(
            engine,
            b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    numerics::assert_slices_close(
        "cached vs direct contract",
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        terms,
    );
}

fn assert_pair_case<R, D>(
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
    D: TensorScalar + Send + Sync + numerics::Numeric + 'static,
{
    let lhs = TensorMap::<R, D>::rand_with_seed(engine, runtime, [space], [space], seed).unwrap();
    let rhs =
        TensorMap::<R, D>::rand_with_seed(engine, runtime, [space], [space], seed + 1).unwrap();
    assert_pair(engine, &lhs, &rhs);
}

#[test]
fn typed_host_network_provider_dtype_matrix_matches_direct_contract() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = u1_space();
    assert_pair_case::<_, f64>(engine, &runtime, &u1, 750_100);
    assert_pair_case::<_, Complex64>(engine, &runtime, &u1, 750_102);

    let su2 = su2_space();
    assert_pair_case::<_, f64>(engine, &runtime, &su2, 750_110);
    assert_pair_case::<_, Complex64>(engine, &runtime, &su2, 750_112);

    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    assert_pair_case::<_, f64>(engine, &runtime, &fz2, 750_120);
    assert_pair_case::<_, Complex64>(engine, &runtime, &fz2, 750_122);

    let product = GradedSpace::try_new(
        Arc::new(
            symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
                .unwrap(),
        ),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(0)), 1),
        ],
    )
    .unwrap();
    assert_pair_case::<_, f64>(engine, &runtime, &product, 750_130);
    assert_pair_case::<_, Complex64>(engine, &runtime, &product, 750_132);

    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 1, "one structural topology is shared");
    assert_eq!(
        stats.workspaces_created, 8,
        "each (provider, dtype) pair has its own downcast-safe Host pool"
    );
    assert!(
        stats.idle_workspaces <= 2,
        "the plan-wide idle budget is shared by all typed pools"
    );
}

#[test]
fn typed_workspace_pool_registry_evicts_cold_types_and_recreates_them() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = u1_space();
    assert_pair_case::<_, f64>(engine, &runtime, &u1, 750_200);
    assert_pair_case::<_, Complex64>(engine, &runtime, &u1, 750_202);

    let su2 = su2_space();
    assert_pair_case::<_, f64>(engine, &runtime, &su2, 750_210);

    // The third typed pool evicts the first from the bounded registry. A later
    // visit must create a fresh pool without growing the plan-wide idle budget.
    assert_pair_case::<_, f64>(engine, &runtime, &u1, 750_230);

    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 1);
    assert_eq!(stats.workspaces_created, 4);
    assert_eq!(stats.workspace_reuses, 0);
    assert!(stats.idle_workspaces <= 2);
}

fn assert_high_rank_pairwise<R>(engine: &mut dyn Engine, runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let lhs =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space, space], [space, space], 101)
            .unwrap();
    let rhs =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space, space], [space, space], 102)
            .unwrap();
    let actual = net(
        &[op(&["i", "j"], &["k", "l"]), op(&["k", "l"], &["m", "n"])],
        &["i", "j"],
        &["m", "n"],
    )
    .contract(engine, &[&lhs, &rhs])
    .unwrap();
    let expected = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap();
    assert_close(
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        1e-12,
    );
    assert_eq!(actual.codomain_rank(), 2);
    assert_eq!(actual.domain_rank(), 2);
}

#[test]
fn pairwise_network_matches_direct_high_rank_contract() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_high_rank_pairwise(engine, &runtime, &u1_space());
    assert_high_rank_pairwise(engine, &runtime, &su2_space());
}

fn assert_permuted_output<R>(engine: &mut dyn Engine, runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let lhs =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space, space], [space, space], 111)
            .unwrap();
    let rhs =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space, space], [space, space], 112)
            .unwrap();
    let actual = net(
        &[op(&["i", "j"], &["k", "l"]), op(&["k", "l"], &["m", "n"])],
        &["j", "i"],
        &["m", "n"],
    )
    .contract(engine, &[&lhs, &rhs])
    .unwrap();
    let expected = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &[1, 0],
                domain: &[2, 3],
            },
        )
        .unwrap();
    assert_close(
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        1e-12,
    );
}

#[test]
fn permuted_output_labels_match_ordered_contract() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_permuted_output(engine, &runtime, &u1_space());
    assert_permuted_output(engine, &runtime, &su2_space());
}

fn assert_single_permute<R>(engine: &mut dyn Engine, runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let tensor =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space], 121).unwrap();
    let actual = net(&[op(&["i"], &["j"])], &["j"], &["i"])
        .contract(engine, &[&tensor])
        .unwrap();
    let expected = tensor.permute(engine, &[1], &[0]).unwrap();
    assert_close(
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        1e-12,
    );
}

#[test]
fn single_tensor_network_is_a_permute() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_single_permute(engine, &runtime, &u1_space());
    assert_single_permute(engine, &runtime, &su2_space());
}

fn assert_conj_norm<R>(engine: &mut dyn Engine, runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let tensor =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space, space], [space, space], 131)
            .unwrap();
    let actual = net(
        &[
            conj(op(&["i", "j"], &["k", "l"])),
            op(&["i", "j"], &["k", "l"]),
        ],
        &[],
        &[],
    )
    .contract(engine, &[&tensor, &tensor])
    .unwrap()
    .scalar(engine)
    .unwrap();
    let norm = tensor.norm(engine, 2.0).unwrap();
    assert!((actual - norm * norm).abs() <= 1e-10 * (1.0 + norm * norm));
}

#[test]
fn scalar_output_with_conj_matches_norm_squared() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_conj_norm(engine, &runtime, &u1_space());
    assert_conj_norm(engine, &runtime, &su2_space());
}

fn assert_three_tensor_chain<R>(engine: &mut dyn Engine, runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let psi =
        TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space, space], 141).unwrap();
    let h = TensorMap::<R, f64>::rand_with_seed(engine, runtime, [space], [space], 142).unwrap();
    let actual = net(
        &[
            conj(op(&["p"], &["l", "r"])),
            op(&["p"], &["q"]),
            op(&["q"], &["l", "r"]),
        ],
        &[],
        &[],
    )
    .contract(engine, &[&psi, &h, &psi])
    .unwrap()
    .scalar(engine)
    .unwrap();
    let h_psi = h
        .contract(
            engine,
            &psi,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1, 2],
            },
        )
        .unwrap();
    let manual = psi
        .adjoint(engine)
        .unwrap()
        .contract(
            engine,
            &h_psi,
            &ContractSpec {
                lhs: &[2, 0, 1],
                rhs: &[0, 1, 2],
                codomain: &[],
                domain: &[],
            },
        )
        .unwrap()
        .scalar(engine)
        .unwrap();
    assert!((actual - manual).abs() <= 1e-10 * (1.0 + manual.abs()));
}

#[test]
fn three_tensor_chain_with_conj_matches_manual_contraction() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    assert_three_tensor_chain(engine, &runtime, &u1_space());
    assert_three_tensor_chain(engine, &runtime, &su2_space());
}

#[test]
fn wrong_input_codomain_split_is_rejected() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let space = u1_space();
    let lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 161)
            .unwrap();
    let rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 162)
            .unwrap();
    let result = net(&[op(&["i", "j"], &[]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[&lhs, &rhs]);
    assert!(
        matches!(result, Err(Error::InvalidArgument(message)) if message.contains("operand 0 puts"))
    );
}

/// Operands of two Runtimes are refused with the typed `RuntimeMismatch`
/// every other facade operation returns, both by the eager preflight and by
/// a plan executed on operands it was not planned with.
#[test]
fn operands_of_different_runtimes_are_a_runtime_mismatch() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let other = Runtime::builder().build().unwrap();
    let space = u1_space();
    let lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 165)
            .unwrap();
    let rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &runtime, [&space], [&space], 166)
            .unwrap();
    let foreign =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(engine, &other, [&space], [&space], 167)
            .unwrap();
    let network = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"]);
    assert!(matches!(
        network.contract(engine, &[&lhs, &foreign]),
        Err(Error::RuntimeMismatch)
    ));
    let planned = network
        .plan(&[&lhs, &rhs], &symtn_network::GreedyDenseOptimizer)
        .unwrap();
    assert!(matches!(
        planned.execute(engine, &[&lhs, &foreign], &mut Default::default()),
        Err(Error::RuntimeMismatch)
    ));
}

#[test]
fn contracted_leg_degeneracy_mismatch_spells_out_both_legs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let lhs_space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let rhs_space = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 4),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let lhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &runtime,
        [&lhs_space],
        [&lhs_space],
        163,
    )
    .unwrap();
    let rhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &runtime,
        [&rhs_space],
        [&rhs_space],
        164,
    )
    .unwrap();
    let message = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(engine, &[&lhs, &rhs])
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("space mismatch for contracted label `j`"),
        "{message}"
    );
    assert!(message.contains("operand 0 leg 1"), "{message}");
    assert!(message.contains("operand 1 leg 0"), "{message}");
}

/// #1372: the Host `Network::contract` pairwise step is the typed `contract`, so a
/// non-symmetric (unbraided or anyonic) rule is rejected with its one error
/// even for the canonical, crossing-free network, which is exactly `compose`
/// (still admitted).
fn assert_host_network_contraction_rejects_non_symmetric<const ANYONIC: bool>(
    engine: &mut dyn Engine,
) {
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(braiding_probe::RealBraidingProbe::<ANYONIC>),
        [(braiding_probe::ProbeSector, 2)],
    )
    .unwrap();
    let lhs =
        TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&leg], [&leg], 1_372_000).unwrap();
    let rhs =
        TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&leg], [&leg], 1_372_001).unwrap();
    let error = net(&[op(&["a"], &["k"]), op(&["k"], &["b"])], &["a"], &["b"])
        .contract(engine, &[&lhs, &rhs])
        .unwrap_err();
    assert!(
        matches!(
            &error,
            Error::Operation(operation)
                if matches!(
                    **operation,
                    symtn::typed::OperationError::UnsupportedTensorContractScope {
                        message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
                    }
                )
        ),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        lhs.contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .unwrap_err()
        .to_string()
    );
    assert!(lhs.compose(engine, &rhs).is_ok());
}

#[test]
fn host_network_contraction_requires_symmetric_braiding() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    assert_host_network_contraction_rejects_non_symmetric::<false>(engine);
    assert_host_network_contraction_rejects_non_symmetric::<true>(engine);
}
