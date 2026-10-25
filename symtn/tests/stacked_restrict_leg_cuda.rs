//! `StackedTensorMap::restrict_leg` on a real device (#1503).
//!
//! `cuda_transfer_stats` is thread-local. Run with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test integration stacked_restrict_leg_cuda:: -- --ignored`.

#![cfg(feature = "cuda")]

use num_complex::Complex64;

use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, LegSelection, StackedTensorMap, TensorMap};

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

/// One element-table upload and one gather, nothing else.
const ONE_GATHER: (u64, u64, u64, u64, u64) = (1, 0, 1, 0, 0);

macro_rules! device_restrict {
    ($label:expr, $leg:expr) => {{
        let leg = $leg;
        let a = leg(0);
        let dual = a.try_dual().unwrap();
        let other = leg(1);
        let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
        device_restrict!(@dtype $label, &runtime, &a, &dual, &other, f64);
        device_restrict!(@dtype $label, &runtime, &a, &dual, &other, Complex64);
    }};
    (@dtype $label:expr, $runtime:expr, $a:expr, $dual:expr, $other:expr, $d:ty) => {{
        // Why a closure: each expansion becomes its own stack frame. Inlined
        // into one test function, the four symmetries times two dtypes
        // overflow a debug test thread's stack once checked Generic SU(3) is
        // enabled.
        let check = || {
        // B = 3, 17 and 64: the submission count must not move with B.
        for count in [3usize, 17, 64] {
            let members = mixed_members!($runtime, $a, $dual, $d, count);
            let device = StackedTensorMap::pack(&members).unwrap().to_cuda().unwrap();
            for (axis, parent) in [$a, $dual, $a, $dual].into_iter().enumerate() {
                for (choice, selection) in selections(parent).iter().enumerate() {
                    let label = format!(
                        "{} {} B={count} axis {axis} selection {choice}",
                        $label,
                        stringify!($d)
                    );
                    let before = cuda_transfer_stats();
                    let restricted = device.restrict_leg(&[(axis, selection)]).unwrap();
                    assert_eq!(delta(before), ONE_GATHER, "{label}: submissions");
                    assert_eq!(restricted.len(), count, "{label}");

                    // A second restriction reads the `[L', B]` gather output,
                    // and a restriction of a selection reads `[L, B']`.
                    let next = (axis + 2) % 4;
                    let second = &selections([$a, $dual][next % 2])[1];
                    let before = cuda_transfer_stats();
                    let twice = restricted.restrict_leg(&[(next, second)]).unwrap();
                    assert_eq!(delta(before), ONE_GATHER, "{label}: chained submissions");
                    let picked = [count - 1, 0, 1];
                    let of_selected = device
                        .select(&picked)
                        .unwrap()
                        .restrict_leg(&[(axis, selection)])
                        .unwrap();

                    let restricted = restricted.to_host().unwrap();
                    let twice = twice.to_host().unwrap();
                    let of_selected = of_selected.to_host().unwrap();
                    for (i, member) in members.iter().enumerate() {
                        let expected = member.restrict_leg(&[(axis, selection)]).unwrap();
                        let actual = restricted.member(i).unwrap();
                        assert!(
                            actual.structure_signature() == expected.structure_signature(),
                            "{label}: member {i} space"
                        );
                        fixtures::assert_bit_exact(actual.dense_data().unwrap(), expected.dense_data().unwrap(), &label);
                        fixtures::assert_bit_exact(
                            twice.member(i).unwrap().dense_data().unwrap(),
                            expected.restrict_leg(&[(next, second)]).unwrap().dense_data().unwrap(),
                            &label,
                        );
                    }
                    for (j, &i) in picked.iter().enumerate() {
                        fixtures::assert_bit_exact(
                            of_selected.member(j).unwrap().dense_data().unwrap(),
                            members[i].restrict_leg(&[(axis, selection)]).unwrap().dense_data().unwrap(),
                            &label,
                        );
                    }
                }
            }

            // A multi-axis set is one upload and one gather too (#1561),
            // bit-exact to one-axis restrictions applied in turn.
            let (on_a, on_dual) = (selections($a), selections($dual));
            let sets: [Vec<(usize, &LegSelection<_>)>; 3] = [
                vec![(2, &on_a[1]), (0, &on_a[0]), (3, &on_dual[0]), (1, &on_dual[1])],
                vec![(3, &on_dual[1]), (0, &on_a[0])],
                // Non-contiguous kept positions on a codomain and a domain
                // leg (#2095): the Host-built element table carries the
                // gather, so still one upload and one gather.
                vec![(1, &on_dual[2]), (2, &on_a[2])],
            ];
            for (choice, set) in sets.iter().enumerate() {
                let label = format!("{} {} B={count} set {choice}", $label, stringify!($d));
                let before = cuda_transfer_stats();
                let restricted = device.restrict_leg(set).unwrap();
                assert_eq!(delta(before), ONE_GATHER, "{label}: submissions");
                let restricted = restricted.to_host().unwrap();
                for (i, member) in members.iter().enumerate() {
                    let sequential = set.iter().fold(member.clone(), |tensor, &pair| {
                        tensor.restrict_leg(&[pair]).unwrap()
                    });
                    let actual = restricted.member(i).unwrap();
                    assert!(
                        actual.structure_signature() == sequential.structure_signature(),
                        "{label}: member {i} space"
                    );
                    fixtures::assert_bit_exact(actual.dense_data().unwrap(), sequential.dense_data().unwrap(), &label);
                }
            }

            // Rejections are eager's, and submit nothing.
            let on_a = &selections($a)[0];
            let on_dual = &selections($dual)[0];
            let on_other = &selections($other)[0];
            let before = cuda_transfer_stats();
            for (axis, selection) in [(4, on_a), (0, on_dual), (1, on_a), (2, on_other)] {
                let expected = members[0].restrict_leg(&[(axis, selection)]).err().unwrap();
                let actual = device.restrict_leg(&[(axis, selection)]).err().unwrap();
                assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "axis {axis}");
            }
            for set in [vec![], vec![(0, on_a), (0, on_a)]] {
                let expected = members[0].restrict_leg(&set).err().unwrap();
                let actual = device.restrict_leg(&set).err().unwrap();
                assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "set {set:?}");
            }
            assert_eq!(delta(before), (0, 0, 0, 0, 0), "rejections submit nothing");
        }
        };
        check();
    }};
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_restrict_leg_is_bit_exact_with_one_gather_per_call() {
    for_each_symmetry!(device_restrict);

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
    let selection = LegSelection::try_new(&charged, [(q(1), 1..2)]).unwrap();
    let before = cuda_transfer_stats();
    let restricted = device.restrict_leg(&[(0, &selection)]).unwrap();
    assert_eq!(
        delta(before).2,
        0,
        "a blockless structure launches no gather"
    );
    assert_eq!(restricted.len(), 2);
    assert!(restricted
        .to_host()
        .unwrap()
        .member(1)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());

    // Both empty-producing paths keep the `[L, B]` shape the other consumes:
    // a restriction of a blockless selection, and a selection of a blockless
    // restriction, succeed as eager does and launch nothing.
    let expected = empty.restrict_leg(&[(0, &selection)]).unwrap();
    let before = cuda_transfer_stats();
    let of_selected = device
        .select(&[1, 0, 1])
        .unwrap()
        .restrict_leg(&[(0, &selection)])
        .unwrap();
    let of_restricted = restricted.select(&[0]).unwrap();
    assert_eq!(delta(before).2, 0, "blockless chains launch no gather");
    assert_eq!((of_selected.len(), of_restricted.len()), (3, 1));
    for stack in [of_selected, of_restricted] {
        let host = stack.to_host().unwrap();
        assert!(*host.signature() == expected.structure_signature());
        for member in 0..host.len() {
            assert!(host
                .member(member)
                .unwrap()
                .dense_data()
                .unwrap()
                .is_empty());
        }
    }
}
