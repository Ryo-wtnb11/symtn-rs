//! A lazy adjoint's factorization allocates no more payload than
//! materializing the adjoint and factoring that. The parent-reading
//! stages (`svd_vals`, SVD, `pinv`, LQ) save the `O(stored_len)` input copy; a
//! redirect trades it for one copy of each adjointed output, plus the
//! adjoint factor spaces, which are metadata. So the excess over
//! materialize-then-factor, if any, must not grow with the degeneracies.
//! Warm measurements, one dense thread.

use std::hint::black_box;
use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::counting_alloc;

/// Warm `f` once, then return its allocations.
fn warm_allocs(mut f: impl FnMut()) -> counting_alloc::Allocs {
    f();
    counting_alloc::measure(f).1
}

/// Warm `f` once, then return its allocated bytes.
fn warm_bytes(f: impl FnMut()) -> i64 {
    warm_allocs(f).bytes as i64
}

/// Allocation calls of `op` on `lazy` and of materialize-then-`op`.
macro_rules! calls {
    ($engine:ident, $lazy:expr, |$t:ident| $op:expr) => {{
        let lazy = &$lazy;
        let lazy_calls = warm_allocs(|| {
            let $t = lazy;
            black_box($op);
        })
        .calls;
        let eager_calls = warm_allocs(|| {
            let materialized = lazy.materialize($engine).unwrap();
            let $t = &materialized;
            black_box($op);
        })
        .calls;
        (lazy_calls, eager_calls)
    }};
}

/// Bytes of `op` on `lazy` minus those of materialize-then-`op`.
macro_rules! excess {
    ($engine:ident, $lazy:expr, |$t:ident| $op:expr) => {{
        let lazy = &$lazy;
        let lazy_bytes = warm_bytes(|| {
            let $t = lazy;
            black_box($op);
        });
        let eager_bytes = warm_bytes(|| {
            let materialized = lazy.materialize($engine).unwrap();
            let $t = &materialized;
            black_box($op);
        });
        (lazy_bytes, eager_bytes)
    }};
}

/// Every op on the lazy adjoint of a random `[v, v] <- [v, v]` complex
/// tensor, for `v` at degeneracy scale 1 and 3.
macro_rules! assert_excess_is_metadata {
    ($engine:ident, $mode:expr, $runtime:expr, $leg:expr) => {{
        let mut rows = Vec::new();
        let mut payload_bytes = Vec::new();
        for scale in [1usize, 3] {
            let leg = $leg(scale);
            let parent = TensorMap::<_, Complex64>::rand_with_seed(
                $engine,
                &$runtime,
                [&leg, &leg],
                [&leg, &leg],
                7,
            )
            .unwrap();
            payload_bytes.push(
                (parent.dense_data().unwrap().len() * std::mem::size_of::<Complex64>()) as i64,
            );
            let lazy = parent.adjoint($engine).unwrap();
            let (r, c) = (&[0, 1][..], &[2, 3][..]);
            rows.push([
                (
                    "svd_vals",
                    excess!($engine, lazy, |t| t.svd_vals($engine, r, c).unwrap()),
                ),
                (
                    "svd_compact",
                    excess!($engine, lazy, |t| t.svd_compact($engine, r, c).unwrap()),
                ),
                (
                    "svd_full",
                    excess!($engine, lazy, |t| t.svd_full($engine, r, c).unwrap()),
                ),
                (
                    "pinv",
                    excess!($engine, lazy, |t| t.pinv($engine, r, c, 1.0e-12).unwrap()),
                ),
                (
                    "qr_compact",
                    excess!($engine, lazy, |t| t.qr_compact($engine, r, c).unwrap()),
                ),
                (
                    "lq_compact",
                    excess!($engine, lazy, |t| t.lq_compact($engine, r, c).unwrap()),
                ),
                (
                    "lq_full",
                    excess!($engine, lazy, |t| t.lq_full($engine, r, c).unwrap()),
                ),
                (
                    "left_null",
                    excess!($engine, lazy, |t| t.left_null($engine, r, c).unwrap()),
                ),
                (
                    "left_polar",
                    excess!($engine, lazy, |t| t.left_polar($engine, r, c).unwrap()),
                ),
                (
                    "inv",
                    excess!($engine, lazy, |t| t.inv($engine, r, c).unwrap()),
                ),
            ]);
            for (name, (lazy_calls, eager_calls)) in [
                (
                    "lq_compact",
                    calls!($engine, lazy, |t| t.lq_compact($engine, r, c).unwrap()),
                ),
                (
                    "lq_full",
                    calls!($engine, lazy, |t| t.lq_full($engine, r, c).unwrap()),
                ),
            ] {
                eprintln!(
                    "{} {name}: lazy/materialize-then-factor calls {lazy_calls}/{eager_calls} \
                     (scale {scale})",
                    $mode
                );
                assert!(
                    lazy_calls <= eager_calls,
                    "{} {name}: more allocation calls than materialize-then-factor",
                    $mode
                );
            }
        }
        // The LQ seam reads the parent in place: materialize-then-LQ
        // pays at least the adjoint copy of the payload on top of it.
        for (scale_rows, &payload) in rows.iter().zip(&payload_bytes) {
            for (name, (lazy, eager)) in scale_rows {
                if name.starts_with("lq_") {
                    assert!(
                        eager - lazy >= payload,
                        "{} {name}: lazy {lazy} bytes, materialize-then-factor {eager}, \
                         payload {payload}",
                        $mode
                    );
                }
            }
        }
        for (small, large) in rows[0].iter().zip(&rows[1]) {
            let (name, (small_lazy, small_eager)) = small;
            let (_, (large_lazy, large_eager)) = large;
            eprintln!(
                "{} {name}: lazy/materialize-then-factor bytes {small_lazy}/{small_eager} \
                 (scale 1), {large_lazy}/{large_eager} (scale 3)",
                $mode
            );
            let small_excess = (small_lazy - small_eager).max(0);
            assert!(
                large_lazy - large_eager <= small_excess,
                "{} {name}: the lazy excess grows with the payload",
                $mode
            );
        }
    }};
}

/// Purpose: factorizing a lazy adjoint adds no payload-sized allocation
/// over materializing it first, on every dense provider.
#[test]
fn multiplicity_free_lazy_adjoint_factorizations_add_no_payload_allocation() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = |scale: usize| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            [(0, 2), (1, 2), (2, 1)]
                .map(|(twice, d)| (SU2Irrep::from_twice_spin(twice), d * scale)),
        )
        .unwrap()
    };
    assert_excess_is_metadata!(engine, "MF SU(2)", runtime, leg);
}

/// Purpose: the checked-Generic twin of the contract above.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_lazy_adjoint_factorizations_add_no_payload_allocation() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = |scale: usize| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (vec![2i64, 2], scale),
                (vec![1, 0], scale),
                (vec![0, 0], scale),
            ],
        )
        .unwrap()
    };
    assert_excess_is_metadata!(engine, "checked SU(3)", runtime, leg);
}
