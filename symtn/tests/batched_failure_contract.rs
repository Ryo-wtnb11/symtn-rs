//! The one failure rule of every batched plan (#1757): a failed `execute` or
//! `execute_into` (validation error, signature mismatch, injected backend
//! error) leaves `take_output() == None`, and the next valid call on the same
//! workspace succeeds and equals eager per member. Backend errors are injected
//! through the shared `SpyExecutor`; the oracle is eager on an unspied Runtime
//! over the same runtime-independent member data.

use crate::prepared;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};

use symtn::expert::{
    DefaultDenseExecutor, DenseBackend, DenseDotConfig, DenseError, DenseExecutor,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseTensor, DenseWrite, MatrixOp,
};
use symtn::sector::U1FusionRule;
use symtn::typed::{
    ComposePlan, ContractPlan, ContractSpec, Error, Runtime, StackedTensorMap, TensorMap,
};

use prepared::{members, u1_legs};

include!("common/spy_executor.rs");

const COUNT: usize = 3;

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
    let engine = &mut symtn_dense::CpuEngine::serial();
    assert_eq!(stack.len(), expected.len(), "{label}");
    for (member, expected) in expected.iter().enumerate() {
        let actual = stack.member(engine, member).unwrap();
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
    mut f: impl FnMut(
        &TensorMap<U1FusionRule, f64>,
        &TensorMap<U1FusionRule, f64>,
    ) -> TensorMap<U1FusionRule, f64>,
) -> Vec<Vec<f64>> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (lhs, rhs) = operands(&runtime);
    (0..COUNT)
        .map(|i| {
            f(
                &lhs.member(engine, i).unwrap(),
                &rhs.member(engine, i).unwrap(),
            )
            .dense_data()
            .unwrap()
            .to_vec()
        })
        .collect()
}

#[test]
fn compose_failures_leave_no_output_and_the_next_call_recovers() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let expected = eager_members(|a, b| a.compose(engine, b).unwrap());
    let runtime = Runtime::builder().build().unwrap();
    let (lhs, rhs) = operands(&runtime);
    let plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();

    assert!(matches!(
        plan.execute(engine, &rhs, &lhs, &mut workspace),
        Err(Error::BatchSignatureMismatch { member: None, .. })
    ));
    assert!(workspace.take_output().is_none(), "signature mismatch");
    assert!(workspace.retained_bytes() > 0, "the buffer is kept");
    assert_members(
        "after mismatch",
        plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap(),
        &expected,
    );

    let mut short = short_destination(&runtime, [0, 1]);
    assert!(plan
        .execute_into(engine, &lhs, &rhs, &mut short, &mut workspace)
        .is_err());
    assert!(
        workspace.take_output().is_none(),
        "execute_into validation error"
    );
    plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();
    assert_members("taken", &workspace.take_output().unwrap(), &expected);
}

fn contract_case(spec: &ContractSpec<'_>, label: &str) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let expected = eager_members(|a, b| a.contract(engine, b, spec).unwrap());
    let runtime = Runtime::builder().build().unwrap();
    let (lhs, rhs) = operands(&runtime);
    let plan = ContractPlan::new(engine, &lhs, &rhs, spec).unwrap();
    let mut workspace = plan.workspace().unwrap();
    plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();

    assert!(matches!(
        plan.execute(engine, &rhs, &lhs, &mut workspace),
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
        plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap(),
        &expected,
    );

    let order = [spec.codomain[0], spec.codomain[1]];
    let mut short = short_destination(&runtime, order);
    assert!(plan
        .execute_into(engine, &lhs, &rhs, &mut short, &mut workspace)
        .is_err());
    assert!(
        workspace.take_output().is_none(),
        "{label}: execute_into validation error"
    );
    plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();
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

#[cfg(feature = "cuda")]
mod cuda {
    //! Device forms. Backend faults cannot be injected into the CUDA
    //! provider, so these cover validation and member rejection, the errors
    //! a device call can be driven into from the public API.
    use super::*;
    use crate::prepared::eigh::hermitian_members;
    use symtn::typed::{BatchError, CudaStorage, EighFullPlan, Engine, HermitianTol};

    fn host_members(
        engine: &mut dyn Engine,
        cuda: &mut dyn Engine,
        stack: &StackedTensorMap<U1FusionRule, f64, CudaStorage<f64>>,
    ) -> Vec<Vec<f64>> {
        let host = stack.to_host(cuda).unwrap();
        (0..host.len())
            .map(|i| {
                host.member(engine, i)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .to_vec()
            })
            .collect()
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_compose_and_contract_failures_leave_no_output() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let cuda = &mut crate::common::cuda_engine();
        let expected = eager_members(|a, b| a.compose(engine, b).unwrap());
        let runtime = Runtime::builder().build().unwrap();
        let (lhs, rhs) = operands(&runtime);
        let (lhs, rhs) = (lhs.to_cuda(cuda).unwrap(), rhs.to_cuda(cuda).unwrap());

        let plan = ComposePlan::new(cuda, &lhs, &rhs).unwrap();
        let mut workspace = plan.workspace().unwrap();
        plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
        assert!(plan.execute(cuda, &rhs, &lhs, &mut workspace).is_err());
        assert!(workspace.take_output().is_none(), "compose mismatch");
        let output = plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
        for (actual, expected) in host_members(engine, cuda, output).iter().zip(&expected) {
            assert_close("cuda compose", actual, expected);
        }
        let mut short = short_destination(&runtime, [0, 1]).to_cuda(cuda).unwrap();
        assert!(plan
            .execute_into(cuda, &lhs, &rhs, &mut short, &mut workspace)
            .is_err());
        assert!(workspace.take_output().is_none(), "compose execute_into");

        for (codomain, label) in [([0, 1], "direct"), ([1, 0], "copyC")] {
            let spec = ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &codomain,
                domain: &[2],
            };
            let expected = eager_members(|a, b| a.contract(engine, b, &spec).unwrap());
            let plan = ContractPlan::new(cuda, &lhs, &rhs, &spec).unwrap();
            let mut workspace = plan.workspace().unwrap();
            plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
            assert!(plan.execute(cuda, &rhs, &lhs, &mut workspace).is_err());
            assert!(workspace.take_output().is_none(), "{label} mismatch");
            let output = plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
            for (actual, expected) in host_members(engine, cuda, output).iter().zip(&expected) {
                assert_close(label, actual, expected);
            }
            let mut short = short_destination(&runtime, codomain).to_cuda(cuda).unwrap();
            assert!(plan
                .execute_into(cuda, &lhs, &rhs, &mut short, &mut workspace)
                .is_err());
            assert!(workspace.take_output().is_none(), "{label} execute_into");
            plan.execute(cuda, &lhs, &rhs, &mut workspace).unwrap();
            assert!(workspace.take_output().is_some(), "{label} recovered");
        }
    }

    #[test]
    #[ignore = "requires a real CUDA device"]
    fn cuda_eigh_failures_leave_no_output() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let cuda = &mut crate::common::cuda_engine();
        let (v, w) = u1_legs();
        let runtime = Runtime::builder().build().unwrap();
        let good = hermitian_members(&runtime, &[&v], COUNT, 7);
        let mut bad = good.clone();
        bad[1] = members::<_, f64>(&runtime, &[&v], &[&v], 1, 12).remove(0);
        let good = StackedTensorMap::pack(&good)
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
        let bad = StackedTensorMap::pack(&bad).unwrap().to_cuda(cuda).unwrap();
        let other = StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v], &[&w], 2, 9))
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
        let plan = EighFullPlan::new(cuda, &good, &[0], &[1], HermitianTol::DEFAULT).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let first = {
            let output = plan.execute(cuda, &good, &mut workspace).unwrap();
            (
                host_members(engine, cuda, output.d),
                host_members(engine, cuda, output.v),
            )
        };
        assert!(matches!(
            plan.execute(cuda, &other, &mut workspace).map(|_| ()),
            Err(BatchError::Operation(Error::BatchSignatureMismatch { .. }))
        ));
        assert!(workspace.take_output().is_none(), "signature mismatch");
        plan.execute(cuda, &good, &mut workspace).unwrap();
        assert!(matches!(
            plan.execute(cuda, &bad, &mut workspace).map(|_| ()),
            Err(BatchError::MemberRejected { .. })
        ));
        assert!(workspace.take_output().is_none(), "member rejection");
        let output = plan.execute(cuda, &good, &mut workspace).unwrap();
        assert_eq!(
            (
                host_members(engine, cuda, output.d),
                host_members(engine, cuda, output.v)
            ),
            first
        );
    }
}
