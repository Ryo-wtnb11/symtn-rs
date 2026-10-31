//! Warm Unique-fusion `trace_pairs` / `trace_pairs_into` allocation contract
//! (#2192).

use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::counting_alloc;

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn u1_leg(charges: i32) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-charges..=charges).map(|q| (U1Irrep::new(q), 2)),
    )
    .unwrap()
}

fn fz2u1_leg(charges: i32) -> GradedSpace<Fz2U1> {
    let sector = |q: i32| {
        let parity = if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        };
        ProductSector::new(parity, U1Irrep::new(q))
    };
    GradedSpace::try_new(
        Arc::new(Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap()),
        (-charges..=charges).map(|q| (sector(q), 2)),
    )
    .unwrap()
}

/// `(source blocks, warm trace_pairs calls, warm trace_pairs_into calls)` of
/// `[V^half] ← [V^half]` traced over `pairs`, after three warm-ups.
macro_rules! warm_trace_calls {
    ($engine:ident, $leg:expr, $half:expr, $pairs:expr) => {{
        let runtime = Runtime::builder().build().unwrap();
        let leg = $leg;
        let legs = vec![&leg; $half];
        let source = TensorMap::<_, f64>::rand_with_seed(
            $engine,
            &runtime,
            legs.iter().copied(),
            legs.iter().copied(),
            2192,
        )
        .unwrap();
        let pairs: &[(usize, usize)] = $pairs;
        let mut destination = source.trace_pairs($engine, pairs).unwrap();
        for _ in 0..3 {
            black_box(source.trace_pairs($engine, pairs).unwrap());
            source
                .trace_pairs_into($engine, pairs, &mut destination, 1.0, 0.5)
                .unwrap();
        }
        let (output, owned) =
            counting_alloc::measure(|| black_box(source.trace_pairs($engine, pairs)));
        output.unwrap();
        let (result, into) = counting_alloc::measure(|| {
            source.trace_pairs_into($engine, pairs, black_box(&mut destination), 1.0, 0.5)
        });
        result.unwrap();
        (source.subblock_count(), owned.calls, into.calls)
    }};
}

/// What: a warm Unique-fusion trace makes no allocation per term. Its
/// permuted and split trees live in caller-owned scratch, as TensorKit's
/// isbits trees do (`_trace_permute!(::UniqueFusion)`), and its outputs are
/// sized up front; so the owned and `_into` call counts are the same however
/// many source blocks (hence terms) there are: ranks 4 and 6, one and two
/// traced pairs, a bosonic and a fermionic rule.
#[test]
fn warm_unique_trace_allocations_do_not_grow_with_terms() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let check = |label: &str, small: (usize, u64, u64), large: (usize, u64, u64)| {
        assert!(
            large.0 >= 10 * small.0,
            "{label}: {} vs {} blocks",
            small.0,
            large.0
        );
        assert_eq!(
            (small.1, small.2),
            (large.1, large.2),
            "{label}: warm (owned, into) calls at {} and {} blocks",
            small.0,
            large.0
        );
    };
    for (half, pairs) in [
        (2, &[(0, 2)][..]),
        (3, &[(0, 3)][..]),
        (3, &[(0, 3), (1, 4)][..]),
    ] {
        check(
            "U(1)",
            warm_trace_calls!(engine, u1_leg(1), half, pairs),
            warm_trace_calls!(engine, u1_leg(3), half, pairs),
        );
        check(
            "fZ2xU(1)",
            warm_trace_calls!(engine, fz2u1_leg(1), half, pairs),
            warm_trace_calls!(engine, fz2u1_leg(3), half, pairs),
        );
    }
}
