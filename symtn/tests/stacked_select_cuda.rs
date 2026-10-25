//! `StackedTensorMap::select` on a real device (#1502).
//!
//! `cuda_transfer_stats` is thread-local. Run with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test integration stacked_select_cuda:: -- --ignored`.

#![cfg(feature = "cuda")]

use num_complex::Complex64;

use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::typed::{Error, Runtime};
use symtn::typed::{GradedSpace, StackedTensorMap, TensorMap};

use crate::stacked_fixtures as fixtures;

/// Submissions and transfers since `before`, as
/// `(h2d, d2h, copy, gemm, solver)` calls.
fn delta(before: CudaTransferStats) -> (u64, u64, u64, u64, u64) {
    let now = cuda_transfer_stats();
    (
        now.h2d_calls - before.h2d_calls,
        now.d2h_calls - before.d2h_calls,
        now.copy_calls - before.copy_calls,
        now.gemm_calls - before.gemm_calls,
        now.solver_calls - before.solver_calls,
    )
}

/// One index upload and one gather, nothing else.
const ONE_GATHER: (u64, u64, u64, u64, u64) = (1, 0, 1, 0, 0);

/// Reordering, a subset (B' < B), the identity (B' = B) and duplicates
/// (B' > B), over the first three members.
const SELECTIONS: [&[usize]; 4] = [&[2, 0, 1], &[1], &[0, 1, 2], &[1, 1, 0, 2, 2, 1]];

macro_rules! device_select {
    ($label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        device_select!(@dtype $label, &runtime, &a, f64);
        device_select!(@dtype $label, &runtime, &a, Complex64);
    }};
    (@dtype $label:expr, $runtime:expr, $a:expr, $d:ty) => {{
        // B = 3, 17 and 64: the submission count must not move with B.
        for count in [3usize, 17, 64] {
            let members = members!($runtime, $a, $d, count);
            let device = StackedTensorMap::pack(&members).unwrap().to_cuda().unwrap();
            let mut selections: Vec<Vec<usize>> =
                SELECTIONS.iter().map(|selection| selection.to_vec()).collect();
            selections.push((0..count).rev().collect());
            selections.push((0..2 * count).map(|j| (7 * j) % count).collect());
            for selection in &selections {
                let label = format!("{} {} B={count} {:?}", $label, stringify!($d), selection);
                let before = cuda_transfer_stats();
                let selected = device.select(selection).unwrap();
                assert_eq!(delta(before), ONE_GATHER, "{label}: select submissions");
                assert_eq!(selected.len(), selection.len(), "{label}");
                assert!(*selected.signature() == *device.signature(), "{label}: signature");

                // A selection of a selection reads the `[L, B']` gather output.
                let reversed: Vec<usize> = (0..selection.len()).rev().collect();
                let before = cuda_transfer_stats();
                let again = selected.select(&reversed).unwrap();
                assert_eq!(delta(before), ONE_GATHER, "{label}: nested select submissions");

                let (selected, again) = (selected.to_host().unwrap(), again.to_host().unwrap());
                for (j, &i) in selection.iter().enumerate() {
                    fixtures::assert_bit_exact(
                        selected.member(j).unwrap().dense_data().unwrap(),
                        members[i].dense_data().unwrap(),
                        &label,
                    );
                    fixtures::assert_bit_exact(
                        again.member(selection.len() - 1 - j).unwrap().dense_data().unwrap(),
                        members[i].dense_data().unwrap(),
                        &label,
                    );
                }
            }

            let before = cuda_transfer_stats();
            assert_eq!(
                device.select(&[0, count]).err(),
                Some(Error::BatchMemberOutOfRange { member: count, len: count })
            );
            assert!(matches!(device.select(&[]), Err(Error::InvalidArgument(_))));
            assert_eq!(delta(before), (0, 0, 0, 0, 0), "rejected selections submit nothing");
        }
    }};
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_select_is_bit_exact_with_one_gather_per_call() {
    for_each_symmetry!(device_select);

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let q = fixtures::U1Irrep::new;
    let charged =
        GradedSpace::try_new(std::sync::Arc::new(fixtures::U1FusionRule), [(q(1), 2)]).unwrap();
    let neutral =
        GradedSpace::try_new(std::sync::Arc::new(fixtures::U1FusionRule), [(q(0), 2)]).unwrap();
    let empty = TensorMap::<_, f64>::zeros(&runtime, [&charged], [&neutral]).unwrap();
    let device = StackedTensorMap::pack(&[&empty, &empty])
        .unwrap()
        .to_cuda()
        .unwrap();
    let before = cuda_transfer_stats();
    let selected = device.select(&[1, 0, 1]).unwrap();
    assert_eq!(
        delta(before).2,
        0,
        "a blockless structure launches no gather"
    );
    assert_eq!(selected.len(), 3);
    assert!(selected
        .to_host()
        .unwrap()
        .member(2)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}
