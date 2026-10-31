//! The `cuda` feature is additive: every typed method unified over storage
//! (#1756) has one definition per name, so its path form resolves with and
//! without `cuda`. CI's `cuda-check` compiles this file under `cuda`, where a
//! Host/CUDA twin pair would be `E0034`-ambiguous.
//!
//! Every method unified over storage has a row here.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{
    ComposePlan, ContractPlan, ContractSpec, ContractWorkspace, Direction, Eigh, EighFullPlan,
    GradedSpace, HermitianTol, LegSelection, Qr, Runtime, StackedTensorMap, Svd, TensorMap,
};

#[test]
fn unified_methods_resolve_in_path_form() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let t: TensorMap<_, f64> = TensorMap::isomorphism(engine, &runtime, [&v], [&v]).unwrap();

    // L1 reductions.
    assert_eq!(TensorMap::norm(&t, engine, f64::INFINITY).unwrap(), 1.0);
    assert_eq!(TensorMap::inner(&t, engine, &t).unwrap(), 2.0);

    // L2 elementwise.
    let twice = TensorMap::scale(&t, engine, 2.0).unwrap();
    let sum = TensorMap::axpby(&t, engine, 1.0, &t, 1.0).unwrap();
    assert_eq!(sum.dense_data().unwrap(), twice.dense_data().unwrap());
    let mut zero = TensorMap::zeros_like(&t, engine).unwrap();
    assert_eq!(TensorMap::norm(&zero, engine, 2.0).unwrap(), 0.0);
    TensorMap::axpby_into(&t, engine, &mut zero, 2.0, 0.0).unwrap();
    assert_eq!(zero.dense_data().unwrap(), twice.dense_data().unwrap());

    // L3 compact factors.
    let Svd { s, .. } = TensorMap::svd_compact(&t, engine, &[0], &[1]).unwrap();
    assert!((TensorMap::norm(&s, engine, 2.0).unwrap() - 2f64.sqrt()).abs() < 1e-12);
    let Qr { r, .. } = TensorMap::qr_compact(&t, engine, &[1], &[0]).unwrap();
    assert!((TensorMap::norm(&r, engine, 2.0).unwrap() - 2f64.sqrt()).abs() < 1e-12);

    // L4 eigh_full.
    let Eigh { d, .. } =
        TensorMap::eigh_full(&t, engine, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert!((TensorMap::norm(&d, engine, 2.0).unwrap() - 2f64.sqrt()).abs() < 1e-12);

    // L5 twist.
    let twisted = TensorMap::twist(&t, engine, &[0], Direction::Forward).unwrap();
    assert_eq!(twisted.dense_data().unwrap(), t.dense_data().unwrap());

    // L6 adjoint/materialize.
    let adjoint = TensorMap::adjoint(&t, engine).unwrap();
    let owned = TensorMap::materialize(&adjoint, engine).unwrap();
    assert_eq!(owned.dense_data().unwrap(), t.dense_data().unwrap());
    let view = TensorMap::adjoint_view(&t);
    // `t` is the identity, so `t ∘ t^H` is `t`.
    assert_eq!(
        t.compose(engine, view).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );

    // L7 transforms. `t` is the rank-(1,1) identity, which every one of
    // these leg motions maps to itself.
    let swapped = TensorMap::permute(&t, engine, &[1], &[0]).unwrap();
    assert_eq!(swapped.dense_data().unwrap(), t.dense_data().unwrap());
    let braided = TensorMap::braid(&t, engine, &[1], &[0], &[0, 1]).unwrap();
    assert_eq!(braided.dense_data().unwrap(), t.dense_data().unwrap());
    let transposed = TensorMap::transpose(&t, engine, &[1], &[0]).unwrap();
    let back = TensorMap::repartition(&transposed, engine, 1).unwrap();
    assert_eq!(back.dense_data().unwrap(), transposed.dense_data().unwrap());
    let mut into = TensorMap::zeros_like(&swapped, engine).unwrap();
    TensorMap::permute_into(&t, engine, &[1], &[0], &mut into, 1.0, 0.0).unwrap();
    TensorMap::braid_into(&t, engine, &[1], &[0], &[0, 1], &mut into, 1.0, 1.0).unwrap();
    assert_eq!(into.dense_data().unwrap(), twice.dense_data().unwrap());
    let mut into = TensorMap::zeros_like(&transposed, engine).unwrap();
    TensorMap::transpose_into(&t, engine, &[1], &[0], &mut into, 1.0, 0.0).unwrap();
    TensorMap::repartition_into(&transposed, engine, &mut into, 1.0, 1.0).unwrap();
    assert_eq!(
        into.dense_data().unwrap(),
        TensorMap::scale(&transposed, engine, 2.0)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    // L8 contract. `t` is the identity, so `t · t = t ∘ t = t`.
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    let product = TensorMap::contract(&t, engine, &t, &spec).unwrap();
    assert_eq!(product.dense_data().unwrap(), t.dense_data().unwrap());
    let composed = TensorMap::compose(&t, engine, &t).unwrap();
    assert_eq!(composed.dense_data().unwrap(), t.dense_data().unwrap());
    let mut into = TensorMap::zeros_like(&t, engine).unwrap();
    TensorMap::contract_into(&t, engine, &t, &spec, &mut into, 2.0, 0.0).unwrap();
    assert_eq!(into.dense_data().unwrap(), twice.dense_data().unwrap());

    // L9 trace. The trace of the rank-2 identity is its dimension.
    let traced = TensorMap::trace_pairs(&t, engine, &[(0, 1)]).unwrap();
    assert_eq!(traced.scalar(engine).unwrap(), 2.0);
    let mut into = TensorMap::zeros_like(&traced, engine).unwrap();
    TensorMap::trace_pairs_into(&t, engine, &[(0, 1)], &mut into, 2.0, 0.0).unwrap();
    assert_eq!(into.scalar(engine).unwrap(), 4.0);

    // L10 prepared contract over a one-member stack of `t`.
    let stack = StackedTensorMap::pack(&[&t]).unwrap();
    let mut dst = StackedTensorMap::pack(&[&zero]).unwrap();
    let plan = ContractPlan::new(engine, &stack, &stack, &spec).unwrap();
    let mut workspace = ContractPlan::workspace(&plan).unwrap();
    let out = ContractPlan::execute(&plan, engine, &stack, &stack, &mut workspace).unwrap();
    assert_eq!(
        out.member(engine, 0).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );
    ContractPlan::execute_into(&plan, engine, &stack, &stack, &mut dst, &mut workspace).unwrap();
    assert_eq!(
        dst.member(engine, 0).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );
    assert!(ContractWorkspace::retained_bytes(&workspace) > 0);
    let plan = ComposePlan::new(engine, &stack, &stack).unwrap();
    let mut workspace = ComposePlan::workspace(&plan).unwrap();
    let out = ComposePlan::execute(&plan, engine, &stack, &stack, &mut workspace).unwrap();
    assert_eq!(
        out.member(engine, 0).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );
    ComposePlan::execute_into(&plan, engine, &stack, &stack, &mut dst, &mut workspace).unwrap();
    assert_eq!(
        dst.member(engine, 0).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );
    let taken = ContractWorkspace::take_output(&mut workspace).unwrap();
    assert_eq!(
        taken.member(engine, 0).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );

    // L11 stacked member movement and the prepared eigh.
    let pair = StackedTensorMap::select(&stack, engine, &[0, 0]).unwrap();
    assert_eq!(pair.len(), 2);
    assert_eq!(
        pair.member(engine, 1).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );
    let first = LegSelection::try_new(&v, [(U1Irrep::new(0), 0..1)]).unwrap();
    let corner =
        StackedTensorMap::restrict_leg(&stack, engine, &[(0, &first), (1, &first)]).unwrap();
    assert_eq!(
        corner.member(engine, 0).unwrap().dense_data().unwrap(),
        &[1.0]
    );
    let plan = EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let out = EighFullPlan::execute(&plan, engine, &stack, &mut workspace).unwrap();
    assert_eq!(out.spectra[0][0].values, [1.0, 1.0]);
}
