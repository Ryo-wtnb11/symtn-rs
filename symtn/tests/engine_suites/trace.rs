//! `trace_pairs` against independent oracles: the physical-basis diagonal
//! sum (U(1), SU(2)), the identity contraction (twist-free providers), the
//! trace of the materialized adjoint for a lazy operand, the hand-valued fZ2
//! supertrace, and the sum over producers of a repeated destination.

use super::*;
use crate::contract_cases::{fill, u1, Payload};
use crate::numerics;
use crate::trace_cases::{
    adjoint_pairs, dense_trace, fermion_su2_cases, fermion_u1_cases, su2_cases, u1_cases,
    u1_su2_cases,
};
use num_complex::{Complex32, Complex64};
use std::sync::Arc;
use symtn::sector::{FermionParityFusionRule, U1Irrep, Z2Irrep};
use symtn::typed::{GradedSpace, Runtime};

/// `case` traced on `$backend`, downloaded; then its lazy adjoint traced on
/// `$backend` over the same legs against the Host trace of the materialized
/// adjoint. Evaluates to the owned result.
macro_rules! traced_and_lazy {
    ($backend:ty, $device:ident, $engine:ident, $case:expr) => {{
        let case = $case;
        let source = <$backend>::upload($device, &case.tensor);
        let actual = {
            let traced = source.trace_pairs($device, &case.pairs).unwrap();
            <$backend>::download($device, &traced)
        };
        if let Some(identity) = &case.identity {
            assert_eq!(
                actual.codomain_rank(),
                identity.codomain_rank(),
                "{}",
                case.name
            );
            numerics::assert_nonzero_slices_close(
                case.name,
                actual.dense_data().unwrap(),
                identity.dense_data().unwrap(),
                case.terms(),
            );
        }
        let pairs = adjoint_pairs(&case.tensor, &case.pairs);
        let lazy = source.adjoint($device).unwrap();
        let lazy = {
            let traced = lazy.trace_pairs($device, &pairs).unwrap();
            <$backend>::download($device, &traced)
        };
        let owned = case
            .tensor
            .adjoint($engine)
            .unwrap()
            .materialize($engine)
            .unwrap()
            .trace_pairs($engine, &pairs)
            .unwrap();
        assert_eq!(
            lazy.codomain_rank(),
            owned.codomain_rank(),
            "{} lazy",
            case.name
        );
        numerics::assert_nonzero_slices_close(
            case.name,
            lazy.dense_data().unwrap(),
            owned.dense_data().unwrap(),
            case.terms(),
        );
        actual
    }};
}

macro_rules! every_fixture {
    ($backend:ty, $device:ident, $engine:ident, $runtime:expr, $d:ty) => {{
        let mut oracles = 0;
        for case in &u1_cases::<$d>($runtime) {
            oracles += usize::from(case.dense) + usize::from(case.identity.is_some());
            let actual = traced_and_lazy!($backend, $device, $engine, case);
            if case.dense {
                let physical = actual.to_physical_dense($engine).unwrap().data;
                numerics::assert_nonzero_slices_close(
                    case.name,
                    &physical,
                    &dense_trace(case),
                    case.terms(),
                );
            }
        }
        for case in &su2_cases::<$d>($runtime) {
            oracles += usize::from(case.dense) + usize::from(case.identity.is_some());
            let actual = traced_and_lazy!($backend, $device, $engine, case);
            if case.dense {
                let physical = actual.to_physical_dense($engine).unwrap().data;
                numerics::assert_nonzero_slices_close(
                    case.name,
                    &physical,
                    &dense_trace(case),
                    case.terms(),
                );
            }
        }
        for case in &u1_su2_cases::<$d>($runtime) {
            oracles += usize::from(case.identity.is_some());
            traced_and_lazy!($backend, $device, $engine, case);
        }
        assert_eq!(
            oracles, 12,
            "every bosonic fixture has an independent oracle"
        );
        for case in &fermion_u1_cases::<$d>($runtime) {
            traced_and_lazy!($backend, $device, $engine, case);
        }
        for case in &fermion_su2_cases::<$d>($runtime) {
            traced_and_lazy!($backend, $device, $engine, case);
        }
    }};
}

on_each_engine! {
    /// Every fixture and payload dtype traces to its independent oracle,
    /// owned and through a lazy adjoint.
    fn traces_match_the_independent_oracles_at_every_dtype<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        every_fixture!(B, device, engine, &runtime, f64);
        every_fixture!(B, device, engine, &runtime, Complex64);
        every_fixture!(B, device, engine, &runtime, f32);
        every_fixture!(B, device, engine, &runtime, Complex32);
    }
}

macro_rules! supertrace {
    ($backend:ty, $device:ident, $engine:ident, $runtime:expr, $d:ty) => {{
        let space = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule),
            [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
        )
        .unwrap();
        let tensor: TensorMap<_, $d> =
            TensorMap::from_subblock_fn($engine, $runtime, [&space], [&space], |trees, index| {
                let value = if index[0] != index[1] {
                    9.0
                } else if *trees.coupled() == Z2Irrep::EVEN {
                    2.0 + index[0] as f64
                } else {
                    5.0 + index[0] as f64
                };
                <$d>::entry(value, 0.0)
            })
            .unwrap();
        let source = <$backend>::upload($device, &tensor);
        let traced = source.trace_pairs($device, &[(0, 1)]).unwrap();
        let traced = <$backend>::download($device, &traced);
        numerics::assert_nonzero_slices_close(
            "fZ2 supertrace",
            traced.dense_data().unwrap(),
            &[<$d>::entry(-13.0, 0.0)],
            8,
        );
    }};
}

on_each_engine! {
    /// The fZ2 supertrace of `diag(2, 3 | 5, 6, 7)` with off-diagonal noise
    /// is `(2 + 3) - (5 + 6 + 7) = -13`, where the ordinary sum would be 23.
    fn fz2_trace_is_the_hand_valued_supertrace<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        supertrace!(B, device, engine, &runtime, f64);
        supertrace!(B, device, engine, &runtime, Complex64);
        supertrace!(B, device, engine, &runtime, f32);
        supertrace!(B, device, engine, &runtime, Complex32);
    }
}

on_each_engine! {
    /// Every destination block of `trace(T, (0, 3))` over
    /// `v = (-1:2, 0:1, 1:2)` has one producer per traced sector; the result
    /// is the sum of the traces of `T` restricted to each traced sector, and
    /// two of them overlap on one output entry, so a replay that overwrote
    /// instead of accumulating would fail.
    fn repeated_destinations_accumulate_every_producer<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
        let w = u1(&[(0, 1), (1, 2)]);
        let restricted = |engine: &mut dyn Engine, only: Option<i32>| -> TensorMap<_, f64> {
            let mut next = fill::<_, f64>(61);
            TensorMap::from_subblock_fn(engine, &runtime, [&v, &w, &v], [&v, &w], move |trees, index| {
                let value = next(trees, index);
                match only {
                    Some(charge) if trees.codomain_uncoupled()[0] != U1Irrep::new(charge) => 0.0,
                    _ => value,
                }
            })
            .unwrap()
        };
        let pairs = [(0, 3)];
        let source = B::upload(device, &restricted(engine, None));
        let full = { let traced = source.trace_pairs(device, &pairs).unwrap(); B::download(device, &traced) };
        let parts: Vec<Vec<f64>> = [-1, 0, 1]
            .into_iter()
            .map(|charge| {
                restricted(engine, Some(charge))
                    .trace_pairs(engine, &pairs)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .to_vec()
            })
            .collect();
        let sum: Vec<f64> = (0..full.dense_data().unwrap().len())
            .map(|index| parts.iter().map(|part| part[index]).sum())
            .collect();
        numerics::assert_nonzero_slices_close("sum of producers", full.dense_data().unwrap(), &sum, 64);
        assert!(
            (0..sum.len()).any(|index| parts.iter().filter(|part| part[index] != 0.0).count() >= 2),
            "no output entry has two producers; the fixture proves nothing"
        );
    }
}
