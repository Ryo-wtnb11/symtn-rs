//! The failure rule of the batched Host eigh plan (#1757) under an injected
//! dense fault: a failed `execute` leaves `take_output() == None`, and the
//! next valid call on the same workspace succeeds and equals eager per
//! member. The oracle is eager per member without injection, over the same
//! runtime-independent member data.

use super::*;
use crate::typed::{BatchError, EighFullPlan, StackedTensorMap};

const COUNT: usize = 3;

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let q = U1Irrep::new;
    (
        GradedSpace::try_new(Arc::new(U1FusionRule), [(q(-1), 2), (q(0), 1), (q(1), 3)]).unwrap(),
        GradedSpace::try_new(Arc::new(U1FusionRule), [(q(0), 2), (q(1), 1)]).unwrap(),
    )
}

/// `count` tensors `codomain <- domain` with dyadic entries, distinct per
/// member and per `salt`.
fn members(
    runtime: &Runtime,
    codomain: &[&GradedSpace<U1FusionRule>],
    domain: &[&GradedSpace<U1FusionRule>],
    count: usize,
    salt: usize,
) -> Vec<TensorMap<U1FusionRule, f64>> {
    let engine = &mut crate::typed::CpuEngine::serial();
    (0..count)
        .map(|member| {
            let mut next = 1000 * member + 100 * salt;
            TensorMap::from_subblock_fn(
                engine,
                runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                |_, _| {
                    next += 1;
                    ((next * 37) % 17) as f64 / 8.0 - 1.0
                },
            )
            .unwrap()
        })
        .collect()
}

/// `count` Hermitian tensors `X + X^H` over `codomain <- codomain`, with
/// dyadic entries (so the sum is exactly symmetric), distinct per member.
fn hermitian_members(
    runtime: &Runtime,
    codomain: &[&GradedSpace<U1FusionRule>],
    count: usize,
    salt: usize,
) -> Vec<TensorMap<U1FusionRule, f64>> {
    let engine = &mut crate::typed::CpuEngine::serial();
    members(runtime, codomain, codomain, count, salt)
        .into_iter()
        .map(|x| {
            let adjoint = x.adjoint(engine).unwrap();
            x.axpby(engine, 1.0, &adjoint, 1.0).unwrap()
        })
        .collect()
}

/// A Runtime whose dense work, while the returned guard lives, runs on a spy
/// whose first call to any of `kernels` fails with a backend error.
fn faulty_runtime(kernels: &'static [Kernel]) -> (Runtime, Arc<SpyCounts>, probe::DenseInjection) {
    let counts = Arc::new(SpyCounts::default());
    let spy = SpyExecutor::counting(&counts).failing(kernels, Some(1), "injected backend fault");
    let dense = probe::inject_dense(Box::new(spy));
    (Runtime::builder().build().unwrap(), counts, dense)
}

fn assert_members(label: &str, stack: &StackedTensorMap<U1FusionRule, f64>, expected: &[Vec<f64>]) {
    let engine = &mut crate::typed::CpuEngine::serial();
    assert_eq!(stack.len(), expected.len(), "{label}");
    for (member, expected) in expected.iter().enumerate() {
        let actual = stack.member(engine, member).unwrap();
        let actual = actual.dense_data().unwrap();
        assert_eq!(actual.len(), expected.len(), "{label} member {member}");
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= 1e-12 * (1.0 + e.abs()),
                "{label} member {member}: {a} vs {e}"
            );
        }
    }
}

/// An executor's own `NumericalFailure` is an operation error of the batch,
/// not a member fault: only symtn's nonfinite-eigenvalue check names a
/// member (#1765).
#[test]
fn an_executor_numerical_failure_is_not_a_member_fault() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let (v, _) = u1_legs();
    let counts = Arc::new(SpyCounts::default());
    let spy = SpyExecutor::counting(&counts).failing_numerically(
        Kernel::EIGH,
        Some(1),
        "injected numerical fault",
    );
    let _dense = probe::inject_dense(Box::new(spy));
    let runtime = Runtime::builder().build().unwrap();
    let stack = StackedTensorMap::pack(&hermitian_members(&runtime, &[&v], COUNT, 7)).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    assert!(matches!(
        plan.execute(engine, &stack, &mut workspace).map(|_| ()),
        Err(BatchError::Operation(Error::Operation(error)))
            if matches!(
                *error,
                symtn::typed::OperationError::Dense(DenseError::NumericalFailure { .. })
            )
    ));
    assert_eq!(counts.of(Kernel::EIGH), 1, "the fault was the executor's");
    assert!(workspace.take_output().is_none());
}

#[test]
fn eigh_failures_leave_no_output_and_the_next_call_recovers() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let (v, _) = u1_legs();
    let oracle_runtime = Runtime::builder().build().unwrap();
    let expected: Vec<_> = hermitian_members(&oracle_runtime, &[&v], COUNT, 7)
        .iter()
        .map(|input| {
            let Eigh { d, v } = input
                .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap();
            (
                d.materialize(engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .to_vec(),
                v.dense_data().unwrap().to_vec(),
            )
        })
        .collect();
    let (runtime, counts, _dense) = faulty_runtime(Kernel::EIGH);
    let stack = StackedTensorMap::pack(&hermitian_members(&runtime, &[&v], COUNT, 7)).unwrap();
    let other = StackedTensorMap::pack(&members(&runtime, &[&v], &[&u1_legs().1], 2, 9)).unwrap();
    let plan = EighFullPlan::new(engine, &stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let check = |label: &str,
                 d: &StackedTensorMap<U1FusionRule, f64>,
                 v: &StackedTensorMap<U1FusionRule, f64>| {
        let (ds, vs): (Vec<_>, Vec<_>) = expected.iter().cloned().unzip();
        assert_members(&format!("{label} d"), d, &ds);
        assert_members(&format!("{label} v"), v, &vs);
    };

    assert_eq!(counts.of(Kernel::EIGH), 0);
    assert!(matches!(
        plan.execute(engine, &stack, &mut workspace).map(|_| ()),
        Err(BatchError::Operation(_))
    ));
    assert_eq!(counts.of(Kernel::EIGH), 1, "the fault was the backend's");
    assert!(workspace.take_output().is_none(), "backend error");
    let output = plan.execute(engine, &stack, &mut workspace).unwrap();
    check("after backend error", output.d, output.v);

    assert!(matches!(
        plan.execute(engine, &other, &mut workspace).map(|_| ()),
        Err(BatchError::Operation(Error::BatchSignatureMismatch {
            member: None,
            ..
        }))
    ));
    assert!(workspace.take_output().is_none(), "signature mismatch");
    let bytes = workspace.retained_bytes();
    let output = plan.execute(engine, &stack, &mut workspace).unwrap();
    check("after mismatch", output.d, output.v);
    assert_eq!(
        workspace.retained_bytes(),
        bytes,
        "a rejected stack of another member count keeps the buffers"
    );
    let (d, v) = workspace.take_output().unwrap();
    check("taken", &d, &v);
}
