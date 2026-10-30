//! Device gates of `EighFullPlan`.
//!
//! CUDA transfer counters are thread-local, so these tests run concurrently.
//! Run with `cargo test -p symtn --no-default-features --features cuda,cpu-faer
//! --test integration prepared_eigh_cuda:: -- --ignored`.
//!
//! The device keeps the raw cuSOLVER gauge, and with `B > 1` the batched
//! solver may differ from eager's in the last ULPs, so eigenvectors are
//! checked through gauge-invariant quantities against device eager, the Host
//! eager decomposition and the Jacobi oracle. At `B = 1` the handle must be
//! bit-identical (IEEE `==`) to device eager.

#![cfg(feature = "cuda")]

use crate::common;

use crate::prepared;

use std::collections::BTreeSet;
use std::fmt::Debug;
use symtn::typed::HermitianTol;

use num_complex::{Complex32, Complex64};
use symtn::expert::cuda_transfer_stats;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::Error;
use symtn::typed::{
    BatchError, Eigh, EighFullPlan, EighFullWorkspace, GradedSpace, MemberFault, Runtime,
    StackedTensorMap, TensorMap,
};

use common::DeviceRule;
use prepared::eigh::{
    check_member, degenerate_entry, has_degenerate_group, has_plus_minus_tie, hermitian_members,
    plus_minus_entry, sector_matrices, single_leg,
};
use prepared::{fz2u1_legs, members, su2_legs, u1_legs};

use crate::fixtures;

use fixtures::{codomain_axes, delta, domain_axes};

/// Checks every member of one device execute against device eager (and at
/// `B = 1` bit for bit), and against Host eager through the oracle.
fn check_device_batch<R>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    label: &str,
    inputs: &[TensorMap<R, f64>],
) where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
{
    let _ = (Complex32::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    let device: Vec<_> = inputs.iter().map(|t| t.to_cuda(cuda).unwrap()).collect();
    let stack = StackedTensorMap::pack(inputs)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let plan = EighFullPlan::new(
        engine,
        &stack,
        &codomain_axes(&inputs[0]),
        &domain_axes(&inputs[0]),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut workspace = plan.workspace().unwrap();
    let output = plan.execute(cuda, &stack, &mut workspace).unwrap();
    let d_stack = output.d.to_host(cuda).unwrap();
    let v_stack = output.v.to_host(cuda).unwrap();
    for (member, (input, device_input)) in inputs.iter().zip(&device).enumerate() {
        let what = format!("{label} member {member}/{}", inputs.len());
        let Eigh {
            d: eager_d,
            v: eager_v,
        } = device_input
            .eigh_full(
                cuda,
                &codomain_axes(device_input),
                &domain_axes(device_input),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        let (eager_d, eager_v) = (
            eager_d.to_host(cuda).unwrap(),
            eager_v.to_host(cuda).unwrap(),
        );
        let d = d_stack.member(engine, member).unwrap();
        let v = v_stack.member(engine, member).unwrap();
        assert!(
            d.structure_signature() == eager_d.structure_signature(),
            "{what}: d space"
        );
        assert!(
            v.structure_signature() == eager_v.structure_signature(),
            "{what}: v space"
        );
        if inputs.len() == 1 {
            assert!(
                d.dense_data().unwrap() == eager_d.dense_data().unwrap(),
                "{what}: d bit-identical to eager"
            );
            assert!(
                v.dense_data().unwrap() == eager_v.dense_data().unwrap(),
                "{what}: v bit-identical to eager"
            );
        }
        // Values in `spectra` are the diagonal of `d`, sector for sector.
        let diagonal = sector_matrices(&d);
        for entry in &output.spectra[member] {
            let matrix = &diagonal[&format!("{:?}", entry.sector)];
            let values: Vec<f64> = (0..matrix.rows)
                .map(|i| matrix.data[i + matrix.rows * i])
                .collect();
            assert!(values == entry.values, "{what}: spectra {:?}", entry.sector);
        }
        check_member(&what, input, &d, &v, (&eager_d, &eager_v), f64::EPSILON);
        let Eigh {
            d: host_d,
            v: host_v,
        } = input
            .eigh_full(
                engine,
                &codomain_axes(input),
                &domain_axes(input),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        check_member(
            &format!("{what} vs Host"),
            input,
            &d,
            &v,
            (&host_d, &host_v),
            f64::EPSILON,
        );
    }
}

fn equivalence<R>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    label: &str,
    leg: GradedSpace<R>,
) where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
{
    let runtime = Runtime::builder().build().unwrap();
    for count in [1, 2, 17] {
        let inputs = hermitian_members(&runtime, &[&leg, &leg], count, 3);
        assert!(inputs[0].subblock_count() > 3, "{label}: several blocks");
        check_device_batch(engine, cuda, &format!("{label} B={count}"), &inputs);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_members_equal_device_eager_and_the_oracle() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    equivalence(engine, cuda, "u1", u1_legs().0);
    equivalence(engine, cuda, "su2", su2_legs().0);
    equivalence(engine, cuda, "fz2u1", fz2u1_legs().0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn plus_minus_lambda_and_degenerate_groups_compare_by_value() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_legs().0;
    let j = SU2Irrep::from_twice_spin;
    let su2 =
        GradedSpace::try_new(std::sync::Arc::new(SU2FusionRule), [(j(0), 3), (j(1), 4)]).unwrap();
    for count in [1, 17] {
        let pm = single_leg(&runtime, &leg, count, plus_minus_entry);
        assert!(has_plus_minus_tie(
            &pm[0]
                .eigh_full(
                    engine,
                    &codomain_axes(&pm[0]),
                    &domain_axes(&pm[0]),
                    HermitianTol::DEFAULT
                )
                .unwrap()
                .d
        ));
        check_device_batch(engine, cuda, &format!("u1 ±λ B={count}"), &pm);
        let degenerate = single_leg(&runtime, &su2, count, degenerate_entry);
        assert!(has_degenerate_group(
            &degenerate[0]
                .eigh_full(
                    engine,
                    &codomain_axes(&degenerate[0]),
                    &domain_axes(&degenerate[0]),
                    HermitianTol::DEFAULT
                )
                .unwrap()
                .d
        ));
        check_device_batch(
            engine,
            cuda,
            &format!("su2 degenerate B={count}"),
            &degenerate,
        );
    }
}

/// Per call, independent of `B`: one solver call per coupled sector, one
/// gather plus one copy per coupled sector (every route of this fixture is
/// layout-aligned) and one diagonal copy into `d` per coupled sector, no
/// GEMM, and the same downloads and upload count. A warm call uploads no
/// dense `d`: its bytes
/// are at most the `B · Σ n` eigenvalues, the index tables and the
/// normalizers.
#[test]
#[ignore = "requires a real CUDA device"]
fn submissions_and_transfers_do_not_depend_on_b() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = su2_legs();
    let mut per_b = BTreeSet::new();
    for count in [1, 2, 17] {
        let inputs = hermitian_members(&runtime, &[&leg, &leg], count, 7);
        let sizes: Vec<usize> = sector_matrices(&inputs[0])
            .values()
            .map(|m| m.rows)
            .collect();
        let sectors = sizes.len() as u64;
        let sum_n: usize = sizes.iter().sum();
        let stack = StackedTensorMap::pack(&inputs)
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
        let plan = EighFullPlan::new(
            engine,
            &stack,
            &codomain_axes(&inputs[0]),
            &domain_axes(&inputs[0]),
            HermitianTol::DEFAULT,
        )
        .unwrap();
        let mut workspace = plan.workspace().unwrap();

        let before = cuda_transfer_stats();
        plan.execute(cuda, &stack, &mut workspace).unwrap();
        let cold = delta(cuda_transfer_stats(), before);
        let retained = workspace.retained_bytes();
        assert!(retained > 0, "B={count}: workspace owns capacity");

        let before = cuda_transfer_stats();
        plan.execute(cuda, &stack, &mut workspace).unwrap();
        let warm = delta(cuda_transfer_stats(), before);
        assert_eq!(
            workspace.retained_bytes(),
            retained,
            "B={count}: warm capacity"
        );
        assert_eq!(warm.solver_calls, sectors, "B={count}: solver calls");
        assert_eq!(warm.gemm_calls, 0, "B={count}: no GEMM");
        assert_eq!(
            warm.copy_calls,
            3 * sectors,
            "B={count}: gathers and copies"
        );
        assert_eq!(
            cold.h2d_calls,
            warm.h2d_calls + 2,
            "B={count}: cold allocates d and v"
        );
        let f64_bytes = std::mem::size_of::<f64>();
        let values = count * sum_n * f64_bytes;
        let indices = 2 * count * sum_n * std::mem::size_of::<i64>();
        let normalizers = 2 * sizes.len() * count * f64_bytes;
        assert!(
            warm.h2d_bytes as usize <= values + indices + normalizers,
            "B={count}: warm H2D {} bytes",
            warm.h2d_bytes
        );
        // Exactly symmetric inputs are decided at stage 2 of admission:
        // two admission downloads plus one spectra download.
        assert_eq!(warm.d2h_calls, 3, "B={count}: downloads");
        per_b.insert((
            warm.h2d_calls,
            warm.d2h_calls,
            warm.copy_calls,
            warm.solver_calls,
        ));
    }
    assert_eq!(
        per_b.len(),
        1,
        "per-call counts independent of B: {per_b:?}"
    );
}

fn rejected(result: Result<(), BatchError>) -> Vec<(usize, MemberFault)> {
    match result {
        Ok(()) => Vec::new(),
        Err(BatchError::MemberRejected { members }) => members,
        Err(other) => panic!("unexpected batch error {other:?}"),
    }
}

/// Non-Hermitian members are all named, the solver never launches, and
/// admission downloads three times whatever `B` is.
#[test]
#[ignore = "requires a real CUDA device"]
fn non_hermitian_members_are_named_before_any_solver_launch() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    for count in [3, 17] {
        let mut inputs = hermitian_members(&runtime, &[&leg, &leg], count, 1);
        let x = members::<_, f64>(&runtime, &[&leg, &leg], &[&leg, &leg], 1, 4).remove(0);
        inputs[1] = x.clone();
        inputs[count - 1] = x;
        let stack = StackedTensorMap::pack(&inputs)
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
        let plan = EighFullPlan::new(
            engine,
            &stack,
            &codomain_axes(&inputs[0]),
            &domain_axes(&inputs[0]),
            HermitianTol::DEFAULT,
        )
        .unwrap();
        let mut ws = plan.workspace().unwrap();
        let before = cuda_transfer_stats();
        let got = rejected(plan.execute(cuda, &stack, &mut ws).map(|_| ()));
        let counts = delta(cuda_transfer_stats(), before);
        let fault = MemberFault::NotHermitian;
        assert_eq!(got, vec![(1, fault), (count - 1, fault)]);
        assert_eq!(counts.solver_calls, 0, "no solver launch");
        assert_eq!(counts.gemm_calls, 0, "no assembly");
        assert_eq!(
            counts.d2h_calls, 3,
            "three admission downloads at B={count}"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn mixed_scale_admission_verdicts_equal_device_eager_per_member() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let h = hermitian_members(&runtime, &[&leg, &leg], 1, 5).remove(0);
    let x = members::<_, f64>(&runtime, &[&leg, &leg], &[&leg, &leg], 1, 9).remove(0);
    let skewed = h.axpby(engine, 1.0, &x, 2f64.powi(-30)).unwrap();
    let admitted = h.axpby(engine, 1.0, &x, 2f64.powi(-60)).unwrap();
    let mut inputs = Vec::new();
    for exponent in [0, -1000, 1000, -1060] {
        let s = 2f64.powi(exponent);
        inputs.extend([
            h.scale(engine, s).unwrap(),
            skewed.scale(engine, s).unwrap(),
            admitted.scale(engine, s).unwrap(),
        ]);
    }
    let expected: Vec<_> = inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| {
            input
                .to_cuda(cuda)
                .unwrap()
                .eigh_full(
                    cuda,
                    &codomain_axes(input),
                    &domain_axes(input),
                    HermitianTol::DEFAULT,
                )
                .is_err()
        })
        .map(|(member, _)| (member, MemberFault::NotHermitian))
        .collect();
    assert!(
        expected.len() >= 3 && expected.len() < inputs.len(),
        "{expected:?}"
    );
    let stack = StackedTensorMap::pack(&inputs)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let plan = EighFullPlan::new(
        engine,
        &stack,
        &codomain_axes(&inputs[0]),
        &domain_axes(&inputs[0]),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    assert_eq!(
        rejected(plan.execute(cuda, &stack, &mut ws).map(|_| ())),
        expected
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn non_finite_entries_and_eigenvalues_reject_their_members() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let mut inputs = single_leg(&runtime, &leg, 5, |_, _, _| 1.0);
    inputs[1] = inputs[1].scale(engine, f64::NAN).unwrap();
    let stack = StackedTensorMap::pack(&inputs)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let plan = EighFullPlan::new(
        engine,
        &stack,
        &codomain_axes(&inputs[0]),
        &domain_axes(&inputs[0]),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    assert_eq!(
        rejected(plan.execute(cuda, &stack, &mut ws).map(|_| ())),
        vec![(1, MemberFault::NotHermitian)]
    );

    // Finite, admitted, but `2M` overflows in the solver.
    let huge = f64::MAX / 1.5;
    let mut inputs = single_leg(&runtime, &leg, 5, |_, _, _| 1.0);
    inputs[0] = inputs[0].scale(engine, huge).unwrap();
    inputs[3] = inputs[3].scale(engine, huge).unwrap();
    let eager: Vec<bool> = inputs
        .iter()
        .map(|input| {
            input
                .to_cuda(cuda)
                .unwrap()
                .eigh_full(
                    cuda,
                    &codomain_axes(input),
                    &domain_axes(input),
                    HermitianTol::DEFAULT,
                )
                .is_err()
        })
        .collect();
    assert_eq!(
        eager,
        [true, false, false, true, false],
        "eager rejects the overflow"
    );
    let stack = StackedTensorMap::pack(&inputs)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let plan = EighFullPlan::new(
        engine,
        &stack,
        &codomain_axes(&inputs[0]),
        &domain_axes(&inputs[0]),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let fault = MemberFault::NonFiniteEigenvalue;
    assert_eq!(
        rejected(plan.execute(cuda, &stack, &mut ws).map(|_| ())),
        vec![(0, fault), (3, fault)]
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn complex_device_payloads_are_unsupported() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let complex = members::<_, Complex64>(&runtime, &[&leg], &[&leg], 2, 1);
    let stack = StackedTensorMap::pack(&complex)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    assert!(matches!(
        EighFullPlan::new(engine, &stack, &codomain_axes(&complex[0]), &domain_axes(&complex[0]), HermitianTol::DEFAULT),
        Err(Error::Operation(error)) if matches!(
            *error,
            symtn::typed::OperationError::UnsupportedTensorContractScope { message }
                if message.contains("real payloads")
        )
    ));
}

/// Every member's `d` is exactly zero off its diagonal and carries the
/// member's spectra on it.
fn assert_diagonal_only<R>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    what: &str,
    output: &symtn::typed::EighStackOutput<'_, R, f64, symtn::typed::CudaStorage<f64>>,
) where
    R: DeviceRule,
    <R as symtn::sector::SectorCodec>::Sector: Debug,
{
    let d = output.d.to_host(cuda).unwrap();
    for member in 0..d.len() {
        let matrices = sector_matrices(&d.member(engine, member).unwrap());
        for entry in &output.spectra[member] {
            let m = &matrices[&format!("{:?}", entry.sector)];
            for j in 0..m.cols {
                for i in 0..m.rows {
                    let value = m.data[i + m.rows * j];
                    if i == j {
                        assert!(value == entry.values[i], "{what}: diagonal");
                    } else {
                        assert!(value.to_bits() == 0, "{what}: off-diagonal {value}");
                    }
                }
            }
        }
    }
}

/// After any failed call no output is observable, and the next successful
/// call rewrites every member (reused buffers included): the oracle holds,
/// `d` is zero off its diagonal, and at B = 1 `v` is device eager's bits;
/// the same after `take_output` and after a change of `B`.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_failed_batch_leaves_no_observable_output_and_the_next_call_is_whole() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (leg, _) = u1_legs();
    let good = single_leg(&runtime, &leg, 4, |member, i, j| {
        1.0 + (i + j + member) as f64 / 4.0 + f64::from(u8::from(i == j))
    });
    let mut overflowing = good.clone();
    overflowing[2] = overflowing[2].scale(engine, f64::MAX / 1.5).unwrap();
    let mut skewed = good.clone();
    skewed[1] = members::<_, f64>(&runtime, &[&leg], &[&leg], 1, 3).remove(0);
    let stack = |cuda: &mut dyn symtn::typed::Engine, inputs: &[TensorMap<_, f64>]| {
        StackedTensorMap::pack(inputs)
            .unwrap()
            .to_cuda(cuda)
            .unwrap()
    };
    let plan = EighFullPlan::new(
        engine,
        &stack(cuda, &good),
        &codomain_axes(&good[0]),
        &domain_axes(&good[0]),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let mut ws = plan.workspace().unwrap();
    let check = |engine: &mut dyn symtn::typed::Engine,
                 cuda: &mut dyn symtn::typed::Engine,
                 what: &str,
                 ws: &mut EighFullWorkspace<_, f64, symtn::typed::CudaStorage<f64>>,
                 inputs: &[TensorMap<_, f64>]| {
        let stacked = stack(cuda, inputs);
        let output = plan.execute(cuda, &stacked, ws).unwrap();
        assert_diagonal_only(engine, cuda, what, &output);
        let d = output.d.to_host(cuda).unwrap();
        let v = output.v.to_host(cuda).unwrap();
        for (member, input) in inputs.iter().enumerate() {
            let Eigh {
                d: host_d,
                v: host_v,
            } = input
                .eigh_full(
                    engine,
                    &codomain_axes(input),
                    &domain_axes(input),
                    HermitianTol::DEFAULT,
                )
                .unwrap();
            let (d, v) = (
                d.member(engine, member).unwrap(),
                v.member(engine, member).unwrap(),
            );
            check_member(what, input, &d, &v, (&host_d, &host_v), f64::EPSILON);
            if inputs.len() == 1 {
                let Eigh { v: eager_v, .. } = input
                    .to_cuda(cuda)
                    .unwrap()
                    .eigh_full(
                        cuda,
                        &codomain_axes(input),
                        &domain_axes(input),
                        HermitianTol::DEFAULT,
                    )
                    .unwrap();
                assert!(
                    v.dense_data().unwrap() == eager_v.to_host(cuda).unwrap().dense_data().unwrap(),
                    "{what}: B = 1 bits"
                );
            }
        }
    };
    check(engine, cuda, "first", &mut ws, &good);
    let overflowing_stack = stack(cuda, &overflowing);
    assert!(matches!(
        plan.execute(cuda, &overflowing_stack, &mut ws).map(|_| ()),
        Err(BatchError::MemberRejected { .. })
    ));
    assert!(
        ws.take_output().is_none(),
        "no output after a non-finite rejection"
    );
    check(engine, cuda, "after non-finite", &mut ws, &good);
    let skewed_stack = stack(cuda, &skewed);
    assert!(matches!(
        plan.execute(cuda, &skewed_stack, &mut ws).map(|_| ()),
        Err(BatchError::MemberRejected { .. })
    ));
    assert!(
        ws.take_output().is_none(),
        "no output after a non-Hermitian rejection"
    );
    check(engine, cuda, "after non-Hermitian", &mut ws, &good);
    assert!(ws.take_output().is_some());
    check(engine, cuda, "after take_output", &mut ws, &good);
    check(engine, cuda, "B change", &mut ws, &good[..1]);
}
