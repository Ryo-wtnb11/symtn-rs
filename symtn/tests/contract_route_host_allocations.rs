//! #1859 Host H2: allocation calls of the Host Core and CopyC routes, eager
//! (`contract_into`) and `ContractPlan` members (B = 1, 17), cold and warm,
//! on one thread.
//!
//! Call-count upper bounds recorded on `e6e9cac8` (before Core and CopyC
//! joined the one Host executor); bytes are reported, not asserted.

// The call budgets are measured on the faer provider; a BLAS provider owns
// its own scratch allocations (grouped-GEMM metadata, QR/LQ workspaces), which
// these gates do not model (#2107).
use crate::backend;

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractPlan, ContractSpec, GradedSpace, Runtime, StackedTensorMap, TensorMap};

use crate::counting_alloc;

fn measure(f: impl FnOnce()) -> (u64, u64) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls, allocs.bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Row {
    eager_warm: (u64, u64),
    member1_cold: (u64, u64),
    member1_warm: (u64, u64),
    member17_cold: (u64, u64),
    member17_warm: (u64, u64),
}

fn row(codomain: &[usize]) -> Row {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (U1Irrep::new(-1), 4),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 4),
        ],
    )
    .unwrap();
    let w = GradedSpace::try_new(rule, [(U1Irrep::new(0), 4), (U1Irrep::new(1), 2)]).unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&v], 6).unwrap();
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain,
        domain: &[2],
    };
    let mut out = a.contract(&b, &spec).unwrap();
    a.contract_into(&b, &spec, &mut out, 1.0, 0.0).unwrap();
    let eager_warm = measure(|| a.contract_into(&b, &spec, &mut out, 1.0, 0.0).unwrap());

    let member = |members: usize| {
        let lhs = StackedTensorMap::pack(&vec![&a; members]).unwrap();
        let rhs = StackedTensorMap::pack(&vec![&b; members]).unwrap();
        let plan = ContractPlan::new(&lhs, &rhs, &spec).unwrap();
        let mut workspace = plan.workspace().unwrap();
        plan.execute(&lhs, &rhs, &mut workspace).unwrap();
        let mut dst = workspace.take_output().unwrap();
        let mut workspace = plan.workspace().unwrap();
        let cold = measure(|| {
            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                .unwrap()
        });
        let warm = measure(|| {
            plan.execute_into(&lhs, &rhs, &mut dst, &mut workspace)
                .unwrap()
        });
        (cold, warm)
    };
    let (member1_cold, member1_warm) = member(1);
    let (member17_cold, member17_warm) = member(17);
    Row {
        eager_warm,
        member1_cold,
        member1_warm,
        member17_cold,
        member17_warm,
    }
}

/// `e6e9cac8`, debug test build, one thread.
const BASE: [(&str, Row); 2] = [
    (
        "core",
        Row {
            eager_warm: (11, 1248),
            member1_cold: (9, 328),
            member1_warm: (1, 48),
            member17_cold: (9, 2632),
            member17_warm: (1, 816),
        },
    ),
    (
        "copyC",
        Row {
            eager_warm: (14, 1648),
            member1_cold: (11, 2880),
            member1_warm: (1, 48),
            member17_cold: (13, 45568),
            member17_warm: (1, 816),
        },
    ),
];

#[test]
fn core_and_copy_c_route_allocation_calls_are_bounded_by_base() {
    let _serial = crate::counting_alloc::serial();
    if !backend::faer_is_default_backend() {
        return;
    }
    let rows = [("core", row(&[0, 1])), ("copyC", row(&[1, 0]))];
    for ((name, row), (base_name, base)) in rows.iter().zip(BASE) {
        eprintln!("{name}: {row:?}");
        assert_eq!(*name, base_name);
        // What: no route allocates more calls than its base; bytes are
        // reported only (they move with platform and dependency internals).
        for (head, base) in [
            (row.eager_warm, base.eager_warm),
            (row.member1_cold, base.member1_cold),
            (row.member1_warm, base.member1_warm),
            (row.member17_cold, base.member17_cold),
            (row.member17_warm, base.member17_warm),
        ] {
            assert!(head.0 <= base.0, "{name}: {row:?}");
        }
    }
}
