//! `Network::contract` network execution on single-precision payloads.
//!
//! Admitting `f32`/`Complex32` to `TensorScalar` opens every `D: TensorScalar`
//! bound in `symtn-network` at once, so the cached network path needs its own oracle.
//! It is the same one the typed base suite uses: the `f64`/`Complex64` result
//! of the same network on the widened input, where the twin is filled from one
//! 24-bit stream so the widening is exact.
//!
//! Tolerance is `K * sqrt(n) * eps(f32) * max(1, max|expected|)` with `K = 32`
//! and `n` the payload length of the largest operand — the naive-sum bound for
//! the contraction the network performs.

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Complex32, Complex64};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::network_support;
use network_support::{net, op};

const K: f64 = 32.0;

trait Parts: Copy {
    fn parts(re: f32, im: f32) -> Self;
    fn wide(self) -> Complex64;
}

impl Parts for f32 {
    fn parts(re: f32, _im: f32) -> Self {
        re
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl Parts for f64 {
    fn parts(re: f32, _im: f32) -> Self {
        f64::from(re)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Parts for Complex32 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(re, im)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl Parts for Complex64 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(f64::from(re), f64::from(im))
    }
    fn wide(self) -> Complex64 {
        self
    }
}

fn draw(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 40) as f32) / ((1_u32 << 23) as f32) - 1.0
}

fn draw_parts<D: Parts>(state: &mut u64) -> D {
    let re = draw(state);
    let im = draw(state);
    D::parts(re, im)
}

fn assert_payloads_agree<D: Parts, W: Parts>(what: &str, narrow: &[D], wide: &[W], terms: usize) {
    assert_eq!(narrow.len(), wide.len(), "{what}: payload lengths differ");
    let scale = wide
        .iter()
        .map(|&value| value.wide().norm())
        .fold(0.0f64, f64::max)
        .max(1.0);
    let tolerance = K * (terms as f64).sqrt() * f64::from(f32::EPSILON) * scale;
    for (index, (&got, &expected)) in narrow.iter().zip(wide).enumerate() {
        let error = (got.wide() - expected.wide()).norm();
        assert!(
            error <= tolerance,
            "{what}: entry {index} is {:?} against the widened oracle {:?} \
             (error {error:e} > tolerance {tolerance:e})",
            got.wide(),
            expected.wide(),
        );
    }
}

macro_rules! twin {
    ($engine:ident, $rt:expr, $narrow:ty, $wide:ty, $codomain:expr, $domain:expr, $seed:expr) => {{
        let mut state = $seed;
        let narrow: TensorMap<_, $narrow> =
            TensorMap::from_subblock_fn($engine, $rt, $codomain, $domain, |_, _| {
                draw_parts(&mut state)
            })
            .unwrap();
        let mut state = $seed;
        let wide: TensorMap<_, $wide> =
            TensorMap::from_subblock_fn($engine, $rt, $codomain, $domain, |_, _| {
                draw_parts(&mut state)
            })
            .unwrap();
        (narrow, wide)
    }};
}

macro_rules! network_suite {
    ($suite:ident, $narrow:ty, $wide:ty) => {
        mod $suite {
            use super::*;

            #[test]
            fn u1_chain_matches_the_widened_oracle() {
                let engine = &mut symtn::typed::CpuEngine::serial();
                let runtime = Runtime::builder().build().unwrap();
                let leg = GradedSpace::try_new(
                    Arc::new(U1FusionRule),
                    [
                        (U1Irrep::new(-1), 2),
                        (U1Irrep::new(0), 3),
                        (U1Irrep::new(1), 2),
                    ],
                )
                .unwrap();
                let (a, wa) = twin!(engine, &runtime, $narrow, $wide, [&leg], [&leg], 7_001);
                let (b, wb) = twin!(engine, &runtime, $narrow, $wide, [&leg], [&leg], 7_002);
                let (c, wc) = twin!(engine, &runtime, $narrow, $wide, [&leg], [&leg], 7_003);
                let terms = wa.dense_data().unwrap().len();

                let got = net(
                    &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
                    &["i"],
                    &["l"],
                )
                .contract(engine, &[&a, &b, &c])
                .unwrap();
                let expected = net(
                    &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
                    &["i"],
                    &["l"],
                )
                .contract(engine, &[&wa, &wb, &wc])
                .unwrap();
                assert_payloads_agree(
                    "u1 chain",
                    got.dense_data().unwrap(),
                    expected.dense_data().unwrap(),
                    terms,
                );

                // The plan cache is keyed by structure, not payload dtype; a
                // second execution must therefore still produce the same
                // result at this precision.
                let again = net(
                    &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
                    &["i"],
                    &["l"],
                )
                .contract(engine, &[&a, &b, &c])
                .unwrap();
                assert_eq!(again.dense_data().unwrap(), got.dense_data().unwrap());

                // Full trace, on the single-precision lane.
                let traced = a
                    .trace_pairs(engine, &[(0, 1)])
                    .unwrap()
                    .scalar(engine)
                    .unwrap();
                let wide_traced = wa
                    .trace_pairs(engine, &[(0, 1)])
                    .unwrap()
                    .scalar(engine)
                    .unwrap();
                let tolerance = K
                    * (terms as f64).sqrt()
                    * f64::from(f32::EPSILON)
                    * wide_traced.wide().norm().max(1.0);
                assert!(
                    (traced.wide() - wide_traced.wide()).norm() <= tolerance,
                    "u1 trace: {:?} against {:?}",
                    traced.wide(),
                    wide_traced.wide()
                );
            }

            #[test]
            fn su2_two_tensor_network_matches_the_widened_oracle() {
                let engine = &mut symtn::typed::CpuEngine::serial();
                let runtime = Runtime::builder().build().unwrap();
                let leg = GradedSpace::try_new(
                    Arc::new(SU2FusionRule),
                    [
                        (SU2Irrep::from_twice_spin(0), 2),
                        (SU2Irrep::from_twice_spin(1), 2),
                    ],
                )
                .unwrap();
                let (a, wa) = twin!(
                    engine,
                    &runtime,
                    $narrow,
                    $wide,
                    [&leg, &leg],
                    [&leg],
                    7_101
                );
                let (b, wb) = twin!(
                    engine,
                    &runtime,
                    $narrow,
                    $wide,
                    [&leg],
                    [&leg, &leg],
                    7_102
                );
                let terms = wa.dense_data().unwrap().len();

                let got = net(
                    &[op(&["i", "j"], &["k"]), op(&["k"], &["l", "m"])],
                    &["i", "j"],
                    &["l", "m"],
                )
                .contract(engine, &[&a, &b])
                .unwrap();
                let expected = net(
                    &[op(&["i", "j"], &["k"]), op(&["k"], &["l", "m"])],
                    &["i", "j"],
                    &["l", "m"],
                )
                .contract(engine, &[&wa, &wb])
                .unwrap();
                assert_payloads_agree(
                    "su2 network",
                    got.dense_data().unwrap(),
                    expected.dense_data().unwrap(),
                    terms,
                );
                assert!(
                    got.dense_data()
                        .unwrap()
                        .iter()
                        .any(|&value| value.wide().norm() > 1e-3),
                    "the SU(2) fixture must produce a nonzero result"
                );
            }
        }
    };
}

network_suite!(f32_payload, f32, f64);
network_suite!(complex32_payload, Complex32, Complex64);
