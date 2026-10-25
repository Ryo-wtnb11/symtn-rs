//! The one failure rule of every batched plan (#1757): a failed `execute` or
//! `execute_into` (validation error, signature mismatch, injected backend
//! error) leaves `take_output() == None`, and the next valid call on the same
//! workspace succeeds and equals eager per member. Backend errors are injected
//! through the shared `SpyExecutor`; the oracle is eager on an unspied Runtime
//! over the same runtime-independent member data.

use crate::prepared;

use std::sync::Arc;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};

use symtn::expert::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite, MatrixOp,
};
use symtn::sector::U1FusionRule;
use symtn::typed::{
    BatchError, ComposePlan, ContractPlan, ContractSpec, Eigh, EighFullPlan, Error, HermitianTol,
    Runtime, StackedTensorMap, TensorMap,
};

use prepared::eigh::hermitian_members;
use prepared::{members, u1_legs};

include!("common/spy_executor.rs");

const COUNT: usize = 3;

/// A Runtime whose first call to any of `kernels` fails with a backend error.
fn faulty_runtime(kernels: &'static [Kernel]) -> (Runtime, Arc<SpyCounts>) {
    let counts = Arc::new(SpyCounts::default());
    let spy = SpyExecutor::counting(&counts).failing(kernels, Some(1), "injected backend fault");
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(spy))
        .build()
        .unwrap();
    (runtime, counts)
}

fn assert_close(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}");
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-12 * (1.0 + e.abs()),
            "{label}: {a} vs {e}"
        );
    }
}

fn assert_members(label: &str, stack: &StackedTensorMap<U1FusionRule, f64>, expected: &[Vec<f64>]) {
    assert_eq!(stack.len(), expected.len(), "{label}");
    for (member, expected) in expected.iter().enumerate() {
        let actual = stack.member(member).unwrap();
        assert_close(
            &format!("{label} member {member}"),
            actual.dense_data().unwrap(),
            expected,
        );
    }
}

type Operands = (
    StackedTensorMap<U1FusionRule, f64>,
    StackedTensorMap<U1FusionRule, f64>,
);

/// `lhs: [v, v] <- [w]` and `rhs: [w] <- [v]` stacks of `COUNT` members.
fn operands(runtime: &Runtime) -> Operands {
    let (v, w) = u1_legs();
    (
        StackedTensorMap::pack(&members::<_, f64>(runtime, &[&v, &v], &[&w], COUNT, 1)).unwrap(),
        StackedTensorMap::pack(&members::<_, f64>(runtime, &[&w], &[&v], COUNT, 2)).unwrap(),
    )
}

/// A destination of the wrong member count, rejected before any write.
fn short_destination(
    runtime: &Runtime,
    codomain_order: [usize; 2],
) -> StackedTensorMap<U1FusionRule, f64> {
    let (v, _) = u1_legs();
    let legs = [&v, &v];
    let codomain = [legs[codomain_order[0]], legs[codomain_order[1]]];
    StackedTensorMap::pack(&members::<_, f64>(runtime, &codomain, &[&v], 1, 3)).unwrap()
}

/// Eager per member on an unspied Runtime.
fn eager_members(
    f: impl Fn(
        &TensorMap<U1FusionRule, f64>,
        &TensorMap<U1FusionRule, f64>,
    ) -> TensorMap<U1FusionRule, f64>,
) -> Vec<Vec<f64>> {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (lhs, rhs) = operands(&runtime);
    (0..COUNT)
        .map(|i| {
            f(&lhs.member(i).unwrap(), &rhs.member(i).unwrap())
                .dense_data()
                .unwrap()
                .to_vec()
        })
        .collect()
}

#[test]
fn compose_failures_leave_no_output_and_the_next_call_recovers() {
    let expected = eager_members(|a, b| a.compose(b).unwrap());
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (lhs, rhs) = operands(&runtime);
    let plan = ComposePlan::new(&lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();

    assert!(matches!(
        plan.execute(&rhs, &lhs, &mut workspace),
        Err(Error::BatchSignatureMismatch { member: None, .. })
    ));
    assert!(workspace.take_output().is_none(), "signature mismatch");
    assert!(workspace.retained_bytes() > 0, "the buffer is kept");
    assert_members(
        "after mismatch",
        plan.execute(&lhs, &rhs, &mut workspace).unwrap(),
        &expected,
    );

    let mut short = short_destination(&runtime, [0, 1]);
    assert!(plan
        .execute_into(&lhs, &rhs, &mut short, &mut workspace)
        .is_err());
    assert!(
        workspace.take_output().is_none(),
        "execute_into validation error"
    );
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();
    assert_members("taken", &workspace.take_output().unwrap(), &expected);
}

fn contract_case(spec: &ContractSpec<'_>, label: &str) {
    let expected = eager_members(|a, b| a.contract(b, spec).unwrap());
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let (lhs, rhs) = operands(&runtime);
    let plan = ContractPlan::new(&lhs, &rhs, spec).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();

    assert!(matches!(
        plan.execute(&rhs, &lhs, &mut workspace),
        Err(Error::BatchSignatureMismatch { member: None, .. })
    ));
    assert!(
        workspace.take_output().is_none(),
        "{label}: signature mismatch"
    );
    assert!(
        workspace.retained_bytes() > 0,
        "{label}: the buffer is kept"
    );
    assert_members(
        label,
        plan.execute(&lhs, &rhs, &mut workspace).unwrap(),
        &expected,
    );

    let order = [spec.codomain[0], spec.codomain[1]];
    let mut short = short_destination(&runtime, order);
    assert!(plan
        .execute_into(&lhs, &rhs, &mut short, &mut workspace)
        .is_err());
    assert!(
        workspace.take_output().is_none(),
        "{label}: execute_into validation error"
    );
    plan.execute(&lhs, &rhs, &mut workspace).unwrap();
    assert_members(label, &workspace.take_output().unwrap(), &expected);
}

#[test]
fn contract_failures_leave_no_output_and_the_next_call_recovers() {
    // The direct Core route and the CopyC route (a permuted codomain).
    contract_case(
        &ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2],
        },
        "direct",
    );
    contract_case(
        &ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1, 0],
            domain: &[2],
        },
        "copyC",
    );
}

/// An executor's own `NumericalFailure` is an operation error of the batch,
/// not a member fault: only symtn's nonfinite-eigenvalue check names a
/// member (#1765).
#[test]
fn an_executor_numerical_failure_is_not_a_member_fault() {
    let (v, _) = u1_legs();
    let counts = Arc::new(SpyCounts::default());
    let spy = SpyExecutor::counting(&counts).failing_numerically(
        Kernel::EIGH,
        Some(1),
        "injected numerical fault",
    );
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(spy))
        .build()
        .unwrap();
    let stack = StackedTensorMap::pack(&hermitian_members(&runtime, &[&v], COUNT, 7)).unwrap();
    let plan = EighFullPlan::new(&stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    assert!(matches!(
        plan.execute(&stack, &mut workspace).map(|_| ()),
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
    let (v, _) = u1_legs();
    let oracle_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let expected: Vec<_> = hermitian_members(&oracle_runtime, &[&v], COUNT, 7)
        .iter()
        .map(|input| {
            let Eigh { d, v } = input.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
            (
                d.materialize().unwrap().dense_data().unwrap().to_vec(),
                v.dense_data().unwrap().to_vec(),
            )
        })
        .collect();
    let (runtime, counts) = faulty_runtime(Kernel::EIGH);
    let stack = StackedTensorMap::pack(&hermitian_members(&runtime, &[&v], COUNT, 7)).unwrap();
    let other =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&u1_legs().1], 2, 9)).unwrap();
    let plan = EighFullPlan::new(&stack, &[0], &[1], HermitianTol::DEFAULT).unwrap();
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
        plan.execute(&stack, &mut workspace).map(|_| ()),
        Err(BatchError::Operation(_))
    ));
    assert_eq!(counts.of(Kernel::EIGH), 1, "the fault was the backend's");
    assert!(workspace.take_output().is_none(), "backend error");
    let output = plan.execute(&stack, &mut workspace).unwrap();
    check("after backend error", output.d, output.v);

    assert!(matches!(
        plan.execute(&other, &mut workspace).map(|_| ()),
        Err(BatchError::Operation(Error::BatchSignatureMismatch {
            member: None,
            ..
        }))
    ));
    assert!(workspace.take_output().is_none(), "signature mismatch");
    let bytes = workspace.retained_bytes();
    let output = plan.execute(&stack, &mut workspace).unwrap();
    check("after mismatch", output.d, output.v);
    assert_eq!(
        workspace.retained_bytes(),
        bytes,
        "a rejected stack of another member count keeps the buffers"
    );
    let (d, v) = workspace.take_output().unwrap();
    check("taken", &d, &v);
}

#[cfg(feature = "cuda")]
mod cuda {
    //! Device forms. Backend faults cannot be injected into the CUDA
    //! provider, so these cover validation and member rejection, the errors
    //! a device call can be driven into from the public API.
    use super::*;

    fn device() -> Runtime {
        Runtime::builder().cuda(0).build().unwrap()
    }

    fn host_members(
        stack: &StackedTensorMap<U1FusionRule, f64, symtn::typed::CudaStorage<f64>>,
    ) -> Vec<Vec<f64>> {
        let host = stack.to_host().unwrap();
        (0..host.len())
            .map(|i| host.member(i).unwrap().dense_data().unwrap().to_vec())
            .collect()
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_compose_and_contract_failures_leave_no_output() {
        let expected = eager_members(|a, b| a.compose(b).unwrap());
        let runtime = device();
        let (lhs, rhs) = operands(&runtime);
        let (lhs, rhs) = (lhs.to_cuda().unwrap(), rhs.to_cuda().unwrap());

        let plan = ComposePlan::new(&lhs, &rhs).unwrap();
        let mut workspace = plan.workspace().unwrap();
        plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        assert!(plan.execute(&rhs, &lhs, &mut workspace).is_err());
        assert!(workspace.take_output().is_none(), "compose mismatch");
        let output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        for (actual, expected) in host_members(output).iter().zip(&expected) {
            assert_close("cuda compose", actual, expected);
        }
        let mut short = short_destination(&runtime, [0, 1]).to_cuda().unwrap();
        assert!(plan
            .execute_into(&lhs, &rhs, &mut short, &mut workspace)
            .is_err());
        assert!(workspace.take_output().is_none(), "compose execute_into");

        for (codomain, label) in [([0, 1], "direct"), ([1, 0], "copyC")] {
            let spec = ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &codomain,
                domain: &[2],
            };
            let expected = eager_members(|a, b| a.contract(b, &spec).unwrap());
            let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
            let mut workspace = plan.workspace().unwrap();
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            assert!(plan.execute(&rhs, &lhs, &mut workspace).is_err());
            assert!(workspace.take_output().is_none(), "{label} mismatch");
            let output = plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            for (actual, expected) in host_members(output).iter().zip(&expected) {
                assert_close(label, actual, expected);
            }
            let mut short = short_destination(&runtime, codomain).to_cuda().unwrap();
            assert!(plan
                .execute_into(&lhs, &rhs, &mut short, &mut workspace)
                .is_err());
            assert!(workspace.take_output().is_none(), "{label} execute_into");
            plan.execute(&lhs, &rhs, &mut workspace).unwrap();
            assert!(workspace.take_output().is_some(), "{label} recovered");
        }
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_eigh_failures_leave_no_output() {
        let (v, w) = u1_legs();
        let runtime = device();
        let good = hermitian_members(&runtime, &[&v], COUNT, 7);
        let mut bad = good.clone();
        bad[1] = members::<_, f64>(&runtime, &[&v], &[&v], 1, 12).remove(0);
        let good = StackedTensorMap::pack(&good).unwrap().to_cuda().unwrap();
        let bad = StackedTensorMap::pack(&bad).unwrap().to_cuda().unwrap();
        let other = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&w], 2, 9))
            .unwrap()
            .to_cuda()
            .unwrap();
        let plan = EighFullPlan::new(&good, &[0], &[1], HermitianTol::DEFAULT).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let first = {
            let output = plan.execute(&good, &mut workspace).unwrap();
            (host_members(output.d), host_members(output.v))
        };
        assert!(matches!(
            plan.execute(&other, &mut workspace).map(|_| ()),
            Err(BatchError::Operation(Error::BatchSignatureMismatch { .. }))
        ));
        assert!(workspace.take_output().is_none(), "signature mismatch");
        plan.execute(&good, &mut workspace).unwrap();
        assert!(matches!(
            plan.execute(&bad, &mut workspace).map(|_| ()),
            Err(BatchError::MemberRejected { .. })
        ));
        assert!(workspace.take_output().is_none(), "member rejection");
        let output = plan.execute(&good, &mut workspace).unwrap();
        assert_eq!((host_members(output.d), host_members(output.v)), first);
    }
}
