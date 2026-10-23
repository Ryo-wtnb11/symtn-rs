//! Independent Phase A smoke program for the canonical typed public API.

use symtn::sector::TypedSectorAdmission;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::ContractSpec;
use symtn::typed::{GradedSpace, Runtime, TensorMap, Truncation};
use symtn::typed::{
    Svd, TensorScalar, TypedTensorConstructionDispatch, TypedTensorModeDispatch,
    TypedTensorRootDispatch,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

use symtn::typed::{Complex32, Complex64};

fn inspect_with_the_existing_root_bound<R, D>(tensor: &TensorMap<R, D>)
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    assert_eq!(tensor.subblocks().unwrap().count(), tensor.subblock_count());
    if tensor.subblock_count() != 0 {
        let _ = tensor.subblock_fusion_trees(0).unwrap();
    }
}

fn zeros_with_the_construction_bound<'a, R, D>(
    runtime: &Runtime,
    space: &'a GradedSpace<R>,
) -> Result<TensorMap<R, D>, <R::Mode as TypedTensorModeDispatch<R>>::FacadeError>
where
    R: TypedSectorAdmission + 'a,
    R::Mode: TypedTensorConstructionDispatch<R, D>,
    D: TensorScalar,
{
    TensorMap::zeros(runtime, [space], [space])
}

#[test]
fn constructs_u1_su2_and_product_tensors_from_provider_labels() {
    let runtime = Runtime::builder().build().unwrap();

    let u1 = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    let u1_tensor = TensorMap::<U1FusionRule, f64>::zeros(&runtime, [&u1], [&u1]).unwrap();
    inspect_with_the_existing_root_bound(&u1_tensor);
    let generic_u1: TensorMap<U1FusionRule, f64> =
        zeros_with_the_construction_bound(&runtime, &u1).unwrap();
    assert_eq!(generic_u1.subblock_count(), u1_tensor.subblock_count());
    let coupled: Vec<_> = (0..u1_tensor.subblock_count())
        .map(|block| *u1_tensor.subblock_fusion_trees(block).unwrap().coupled())
        .collect();
    assert!(coupled.contains(&U1Irrep::new(-1)));
    assert!(coupled.contains(&U1Irrep::new(0)));

    let su2 = GradedSpace::try_new(
        std::sync::Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_tensor = TensorMap::<SU2FusionRule, f64>::zeros(&runtime, [&su2], [&su2]).unwrap();
    assert!(su2_tensor.subblock_count() >= 2);

    let product = GradedSpace::try_new(
        std::sync::Arc::new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                FermionParityFusionRule,
                U1FusionRule,
            )
            .unwrap(),
        ),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
        ],
    )
    .unwrap();
    let product_tensor = TensorMap::<_, f64>::zeros(&runtime, [&product], [&product]).unwrap();
    assert_eq!(product_tensor.subblock_count(), 2);
}

#[test]
fn u1_index_contraction_trace_and_decomposition_paths_are_executable() {
    let runtime = Runtime::builder().build().unwrap();
    let space =
        GradedSpace::try_new(std::sync::Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let tensor = TensorMap::<U1FusionRule, f64>::from_subblock_fn(
        &runtime,
        [&space],
        [&space],
        |_, index| match index {
            [0, 0] => 3.0,
            [1, 1] => 2.0,
            _ => 1.0,
        },
    )
    .unwrap();

    let identity =
        TensorMap::<U1FusionRule, f64>::isomorphism(&runtime, [&space], [&space]).unwrap();
    // Each entry sums the two-dimensional contracted leg.
    numerics::assert_slices_close(
        "identity contraction",
        identity
            .contract(
                &tensor,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap()
            .dense_data()
            .unwrap(),
        tensor.dense_data().unwrap(),
        2,
    );
    numerics::assert_slices_close(
        "double adjoint",
        tensor
            .adjoint()
            .unwrap()
            .adjoint()
            .unwrap()
            .dense_data()
            .unwrap(),
        tensor.dense_data().unwrap(),
        1,
    );
    assert_eq!(identity.tr().unwrap(), 2.0);

    let rank_three = TensorMap::<U1FusionRule, f64>::from_subblock_fn(
        &runtime,
        [&space, &space],
        [&space],
        |_, i| (1 + i[0] + 2 * i[1] + 4 * i[2]) as f64,
    )
    .unwrap();
    let roundtrip = rank_three
        .permute(&[1, 0], &[2])
        .unwrap()
        .permute(&[1, 0], &[2])
        .unwrap();
    assert_eq!(
        roundtrip.dense_data().unwrap(),
        rank_three.dense_data().unwrap()
    );

    let Svd { u, s, vh } = tensor.svd_compact(&[0], &[1]).unwrap();
    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), &Truncation::rank(2))
        .unwrap();
    let u = u
        .restrict_leg(&[(u.codomain_rank(), &found.selection)])
        .unwrap();
    let s = s
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let vh = vh.restrict_leg(&[(0, &found.selection)]).unwrap();
    let reconstructed = u.compose(&s).unwrap().compose(&vh).unwrap();
    // Rank-two reconstruction of a rank-two block: two terms per entry.
    numerics::assert_slices_close(
        "truncated SVD reconstruction",
        reconstructed.dense_data().unwrap(),
        tensor.dense_data().unwrap(),
        2,
    );
}

// No "trivial symmetry" smoke is fabricated here: current main exposes no
// canonical no-symmetry provider. The operation matrix classifies that public
// path as unsupported rather than treating a vacuum-only U(1) fixture as one.
