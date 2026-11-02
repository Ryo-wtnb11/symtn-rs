//! General-axes `contract` and `contract_into` against independent oracles:
//! TensorKit's `blas_contract!` step sequence on Host typed ops (bosonic
//! providers), the same sequence with the twist on the B role and on the A
//! role (fermionic providers, with a twist-free control that must differ),
//! the physical-basis dense contraction (U(1), SU(2)), and the
//! TensorKit-valued fZ2 closed loops.

use crate::contract_cases::{
    blas_contract_oracle, dense_oracle, fermionic_blas_contract_oracle, fz2_tensorkit_loops,
    lazy_cases, poisoned_destination, product_general, su2_bent, su2_core_form_lazy_cases,
    su2_reordered, u1_inactive_cases, u1_lhs_identity, u1_rank_five, u1_reordered, u1_rhs_identity,
    Payload, TwistRole,
};
use crate::numerics;
use num_complex::{Complex32, Complex64};
use symtn::typed::Runtime;

/// `$case` contracted on `$backend`, downloaded.
macro_rules! contracted {
    ($backend:ty, $device:ident, $case:expr) => {{
        let case = &$case;
        let lhs = <$backend>::upload($device, &case.lhs);
        let rhs = <$backend>::upload($device, &case.rhs);
        let result = lhs.contract($device, &rhs, &case.spec()).unwrap();
        <$backend>::download($device, &result)
    }};
}

/// `$case` contracted on `$backend` into a NaN-poisoned destination with
/// `alpha = 1, beta = 0`, downloaded.
macro_rules! overwritten {
    ($backend:ty, $device:ident, $engine:ident, $case:expr, $d:ty) => {{
        let case = &$case;
        let poisoned = poisoned_destination($engine, case);
        let mut destination = <$backend>::upload($device, &poisoned);
        // The Host upload shares the payload; the destination must own it.
        drop(poisoned);
        let lhs = <$backend>::upload($device, &case.lhs);
        let rhs = <$backend>::upload($device, &case.rhs);
        lhs.contract_into(
            $device,
            &rhs,
            &case.spec(),
            &mut destination,
            <$d>::entry(1.0, 0.0),
            <$d>::entry(0.0, 0.0),
        )
        .unwrap();
        <$backend>::download($device, &destination)
    }};
}

/// Runs `$body` with `$case` bound to every bosonic fixture: general axes,
/// reordered and bent legs, identity operands on either side, lazy adjoints
/// on either side, and the core-form lazy class.
macro_rules! each_bosonic_case {
    ($engine:ident, $runtime:expr, $d:ty, |$case:ident| $body:block) => {{
        let runtime = $runtime;
        {
            let $case = u1_rank_five::<$d>($engine, runtime);
            $body
        }
        {
            let $case = u1_reordered::<$d>($engine, runtime);
            $body
        }
        {
            let $case = su2_reordered::<$d>($engine, runtime);
            $body
        }
        {
            let $case = su2_bent::<$d>($engine, runtime);
            $body
        }
        {
            let $case = product_general::<$d>($engine, runtime);
            $body
        }
        {
            let $case = u1_lhs_identity::<$d>($engine, runtime);
            $body
        }
        {
            let $case = u1_rhs_identity::<$d>($engine, runtime);
            $body
        }
        let operand = u1_rank_five::<$d>($engine, runtime).lhs;
        for $case in lazy_cases($engine, &operand, "U(1) lazy") {
            $body
        }
        let operand = su2_reordered::<$d>($engine, runtime).lhs;
        for $case in lazy_cases($engine, &operand, "SU(2) lazy") {
            $body
        }
        let operand = product_general::<$d>($engine, runtime).lhs;
        for $case in lazy_cases($engine, &operand, "U(1) x SU(2) lazy") {
            $body
        }
        for $case in su2_core_form_lazy_cases::<$d>($engine, runtime) {
            $body
        }
    }};
}

macro_rules! against_blas {
    ($engine:ident, $actual:expr, $case:expr) => {{
        let (actual, case) = (&$actual, &$case);
        let oracle = blas_contract_oracle($engine, case);
        assert_eq!(
            actual.codomain_rank(),
            oracle.codomain_rank(),
            "{}",
            case.name
        );
        assert_eq!(actual.domain_rank(), oracle.domain_rank(), "{}", case.name);
        numerics::assert_nonzero_slices_close(
            case.name,
            actual.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            case.terms(),
        );
    }};
}

macro_rules! against_twist_roles {
    ($engine:ident, $actual:expr, $case:expr, $twist:expr) => {{
        let (actual, case) = (&$actual, &$case);
        for role in [TwistRole::B, TwistRole::A] {
            let oracle = fermionic_blas_contract_oracle($engine, case, role, $twist);
            numerics::assert_nonzero_slices_close(
                &format!("{} vs the {role:?}-role oracle", case.name),
                actual.dense_data().unwrap(),
                oracle.dense_data().unwrap(),
                case.terms(),
            );
        }
    }};
}

macro_rules! blas_at {
    ($backend:ty, $device:ident, $engine:ident, $runtime:expr, $d:ty) => {{
        each_bosonic_case!($engine, $runtime, $d, |case| {
            let actual = contracted!($backend, $device, case);
            against_blas!($engine, actual, case);
        });
    }};
}

on_each_engine! {
    /// Every bosonic fixture contracts to TensorKit's `blas_contract!`
    /// sequence at every payload dtype.
    fn contract_matches_the_blas_contract_sequence_at_every_dtype<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        blas_at!(B, device, engine, &runtime, f64);
        blas_at!(B, device, engine, &runtime, Complex64);
        blas_at!(B, device, engine, &runtime, f32);
        blas_at!(B, device, engine, &runtime, Complex32);
    }
}

on_each_engine! {
    /// U(1) and SU(2) contractions whose contracted legs do not bend equal
    /// the physical-basis dense contraction.
    fn contract_matches_the_physical_basis_contraction<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        macro_rules! dense {
            ($case:expr) => {{
                let case = $case;
                assert!(case.dense, "{}", case.name);
                let (shape, expected) = dense_oracle(engine, &case);
                let actual = contracted!(B, device, case).to_physical_dense(engine).unwrap();
                assert_eq!(actual.shape, shape, "{}", case.name);
                numerics::assert_nonzero_slices_close(case.name, &actual.data, &expected, case.terms());
            }};
        }
        dense!(u1_reordered::<f64>(engine, &runtime));
        dense!(u1_reordered::<Complex64>(engine, &runtime));
        dense!(su2_reordered::<f64>(engine, &runtime));
        dense!(su2_reordered::<Complex64>(engine, &runtime));
    }
}

macro_rules! fermionic_at {
    ($backend:ty, $device:ident, $engine:ident, $runtime:expr, $d:ty) => {{
        for_each_fermionic_fixture!($engine, $runtime, $d, |case, twist| {
            let actual = contracted!($backend, $device, case);
            against_twist_roles!($engine, actual, case, twist);
            let untwisted = fermionic_blas_contract_oracle($engine, &case, TwistRole::None, twist);
            let scale = actual
                .dense_data()
                .unwrap()
                .iter()
                .map(|value| value.magnitude())
                .fold(0.0_f64, f64::max);
            assert!(
                actual
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(untwisted.dense_data().unwrap())
                    .any(|(&left, &right)| left.distance(right) > 1e-3 * scale),
                "{}: the twist changes nothing here",
                case.name
            );
        });
    }};
}

on_each_engine! {
    /// Every fermionic fixture (general axes with the twist on one or both
    /// contracted legs, canonical forms whose twist varies within a coupled
    /// sector, lazy adjoints on either side) contracts to TensorKit's
    /// sequence with the twist on the B role and on the A role, and not to
    /// the twist-free sequence.
    fn fermionic_contract_matches_both_tensorkit_twist_roles_at_every_dtype<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        fermionic_at!(B, device, engine, &runtime, f64);
        fermionic_at!(B, device, engine, &runtime, Complex64);
        fermionic_at!(B, device, engine, &runtime, f32);
        fermionic_at!(B, device, engine, &runtime, Complex32);
    }
}

on_each_engine! {
    /// The TensorKit-valued fZ2 closed loops as explicit `contract` calls.
    fn fz2_loops_as_explicit_contracts_match_tensorkit<B>() {
        // The helper's closures are `Fn`; each borrows the one engine in turn.
        let device = std::cell::RefCell::new(B::engine());
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let loops = fz2_tensorkit_loops(
            engine,
            &runtime,
            |tensor| B::upload(&mut *device.borrow_mut(), &tensor),
            |_, x, y, spec| x.contract(&mut *device.borrow_mut(), y, spec).unwrap(),
            |engine, tensor| {
                let host = B::download(&mut *device.borrow_mut(), &tensor);
                host.scalar(engine).unwrap()
            },
        );
        for (name, value, expected) in loops {
            assert!((value - expected).abs() < 1e-12, "{name}: {value} vs {expected}");
        }
    }
}

macro_rules! overwrite_at {
    ($backend:ty, $device:ident, $engine:ident, $runtime:expr, $d:ty) => {{
        each_bosonic_case!($engine, $runtime, $d, |case| {
            let written = overwritten!($backend, $device, $engine, case, $d);
            against_blas!($engine, written, case);
        });
        for case in u1_inactive_cases::<$d>($engine, $runtime) {
            let written = overwritten!($backend, $device, $engine, case, $d);
            against_blas!($engine, written, case);
        }
        for_each_fermionic_fixture!($engine, $runtime, $d, |case, twist| {
            let written = overwritten!($backend, $device, $engine, case, $d);
            against_twist_roles!($engine, written, case, twist);
        });
    }};
}

on_each_engine! {
    /// `contract_into` with `beta = 0` rewrites a NaN-poisoned destination
    /// with the independent oracle's values on every fixture, including
    /// destinations with blocks no GEMM writes (by the core route, by an
    /// identity output and under an output transform).
    fn contract_into_a_poisoned_destination_matches_the_oracles_at_every_dtype<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        overwrite_at!(B, device, engine, &runtime, f64);
        overwrite_at!(B, device, engine, &runtime, Complex64);
        overwrite_at!(B, device, engine, &runtime, f32);
        overwrite_at!(B, device, engine, &runtime, Complex32);
    }
}
