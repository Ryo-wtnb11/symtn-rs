//! #1546 performance contract: allocation budget of a borrowed view operand.

use std::hint::black_box;
use std::sync::Arc;
use symtn::typed::ContractSpec;

use num_complex::Complex64;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

fn measure(f: impl FnOnce()) -> (u64, u64) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls, allocs.bytes)
}

fn tensor(runtime: &Runtime, degeneracy: usize, seed: u64) -> TensorMap<U1FusionRule, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-2..=2).map(|charge| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    TensorMap::rand_with_seed(engine, runtime, [&space, &space], [&space], seed).unwrap()
}

/// What: passing `t.adjoint_view()` costs exactly what building the owned
/// lazy adjoint `&t.adjoint()?` and passing it costs (one adjoint header and
/// its logical layout), and that cost does not grow with degeneracy: no
/// payload is copied for the view.
#[test]
fn adjoint_view_operand_costs_the_owned_lazy_adjoint_header() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let mut view_costs = Vec::new();
    for degeneracy in [1, 8] {
        let b = tensor(&runtime, degeneracy, 1);
        let a = tensor(&runtime, degeneracy, 2)
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap();
        black_box(a.inner(engine, b.adjoint_view()).unwrap());
        let view = measure(|| {
            black_box(a.inner(engine, b.adjoint_view()).unwrap());
        });
        let owned = measure(|| {
            let b_adjoint = &b.adjoint(engine).unwrap();
            black_box(a.inner(engine, b_adjoint).unwrap());
        });
        assert_eq!(view, owned, "degeneracy {degeneracy}: inner");
        view_costs.push(view);

        black_box(
            a.contract(
                engine,
                b.adjoint_view(),
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3],
                },
            )
            .unwrap(),
        );
        let contract_view = measure(|| {
            black_box(
                a.contract(
                    engine,
                    b.adjoint_view(),
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                )
                .unwrap(),
            );
        });
        let contract_owned = measure(|| {
            let b_adjoint = &b.adjoint(engine).unwrap();
            black_box(
                a.contract(
                    engine,
                    b_adjoint,
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                )
                .unwrap(),
            );
        });
        assert_eq!(
            contract_view, contract_owned,
            "degeneracy {degeneracy}: contract"
        );
    }
    assert_eq!(
        view_costs[0], view_costs[1],
        "view cost grew with the payload"
    );
}

/// What: a plain view `&t` allocates nothing beyond the operation itself.
#[test]
fn plain_view_operand_allocates_nothing_extra() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let a = tensor(&runtime, 4, 3);
    black_box(a.inner(engine, &a).unwrap());
    let cost = measure(|| {
        black_box(a.inner(engine, &a).unwrap());
    });
    assert_eq!(cost, (0, 0));
}
