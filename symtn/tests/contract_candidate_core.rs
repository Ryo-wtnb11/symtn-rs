//! Zero-copy TensorKit contraction candidates (#1468).
//!
//! TensorKit `contract!` (tensoroperations.jl L318-357) scores the lhs- and
//! rhs-sorted pairings in both operand orders, and
//! `has_shared_permute(::AdjointTensorMap)` (indexmanipulations.jl L282)
//! makes a lazy adjoint free when its permuted order is its natural adjoint
//! order. The probes below are free in TensorKit only after that sort or swap:
//!
//! - C1 `A[3,2]·B[1,0]`, C2 `A[0,1]·B[2,3] → [2,3,0,1]` (owned);
//! - L3 `P'[3,2]·B[1,0]`, L5 `A[0,1]·P'[2,3] → [2,3,0,1]`,
//!   L7 `A[3,2]·P'[1,0]` (`P'` a lazy adjoint).
//!
//! Each must agree with TensorKit's `blas_contract!` step sequence on Host
//! typed operations (which materializes through `permute`, never through
//! the contraction route under test), run no tree transform on a warm call,
//! and allocate no more than its literal Core control (C0 `A[2,3]·B[0,1]`,
//! L4 `P'[2,3]·B[0,1]`).

use crate::contract_cases;
use crate::numerics;

use std::hint::black_box;
use symtn::typed::ContractSpec;
use symtn::typed::Direction;

use contract_cases::{
    blas_contract_oracle, candidate_core_probes as probes, copy_c_probes, dense_oracle, fermion_u1,
    fermionic_blas_contract_oracle, fill, su2, u1, u1_non_self_dual, Case, FermionU1, Payload,
    TwistRole,
};
use num_complex::{Complex32, Complex64};
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::counting_alloc;

/// Values against `oracle`, then the warm route evidence on `f64`.
fn check<R, D>(
    runtime: &Runtime,
    v: &GradedSpace<R>,
    symmetry: &str,
    oracle: impl Fn(&Case<R, D>) -> TensorMap<R, D>,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    for (case, _) in probes::<R, D>(runtime, v) {
        let expected = oracle(&case);
        numerics::assert_nonzero_slices_close(
            &format!("{symmetry} {}", case.name),
            case.host().dense_data().unwrap(),
            expected.dense_data().unwrap(),
            case.terms(),
        );
    }
}

/// Warm `(allocation calls, allocation bytes, tree-transform lookups)`.
fn warm<R>(_runtime: &Runtime, case: &Case<R, f64>) -> (u64, u64, usize)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let call = || {
        black_box(&case.lhs)
            .contract(&case.rhs, &case.spec())
            .unwrap()
    };
    call();
    call();
    let lookups = || {
        let info = completed_transformers();
        (info.hits() + info.misses()) as usize
    };
    let before = lookups();
    let (result, allocs) = counting_alloc::measure(call);
    drop(result);
    (allocs.calls, allocs.bytes, lookups() - before)
}

fn assert_warm_zero_copy<R>(v: &GradedSpace<R>, symmetry: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let probes = probes::<R, f64>(&runtime, v);
    let measured: Vec<_> = probes
        .iter()
        .map(|(case, control)| (case.name, *control, warm(&runtime, case)))
        .collect();
    let control = |name| {
        measured
            .iter()
            .find(|(probe, _, _)| *probe == name)
            .unwrap()
            .2
    };
    for &(name, control_name, (calls, bytes, lookups)) in &measured {
        let (control_calls, control_bytes, _) = control(control_name);
        eprintln!("{symmetry} {name}: {calls} calls, {bytes} B, {lookups} transform lookups");
        // What: the selected candidate transforms neither source nor output.
        assert_eq!(lookups, 0, "{symmetry} {name}: tree transforms ran");
        // What: no more allocation than the literal Core call.
        assert!(
            calls <= control_calls && bytes <= control_bytes,
            "{symmetry} {name}: {calls} calls / {bytes} B vs {control_name} \
             {control_calls} / {control_bytes}"
        );
    }
}

#[test]
fn zero_copy_candidates_run_no_transform_and_allocate_like_literal_core() {
    let _serial = crate::counting_alloc::serial();
    // Its helper serializes: sibling tests read process-global lookup counts.
    assert_warm_zero_copy(&u1_non_self_dual(), "U(1)");
    assert_warm_zero_copy(&su2(), "SU(2)");
    assert_warm_zero_copy(&fermion_u1(), "fZ2xU(1)");
}

fn bosonic_values<D: Payload>() {
    let runtime = Runtime::builder().build().unwrap();
    check::<_, D>(&runtime, &u1_non_self_dual(), "U(1)", blas_contract_oracle);
    check::<_, D>(&runtime, &su2(), "SU(2)", blas_contract_oracle);
}

#[test]
fn zero_copy_candidates_match_the_tensorkit_blas_contract_sequence() {
    let _serial = crate::counting_alloc::serial();
    // Serial: sibling tests read process-global transformer lookup counts.
    bosonic_values::<f64>();
    bosonic_values::<Complex64>();
    bosonic_values::<f32>();
    bosonic_values::<Complex32>();
}

fn fermionic_values<D: Payload>() {
    let runtime = Runtime::builder().build().unwrap();
    let twist =
        |t: &TensorMap<FermionU1, D>, legs: &[usize]| t.twist(legs, Direction::Forward).unwrap();
    // The literal sequence of C2 and L5 contracts B legs that are dual, so
    // `blas_contract!` twists there even though the selected swapped
    // candidate needs no twist; either twist role is the oracle.
    for role in [TwistRole::B, TwistRole::A] {
        check::<_, D>(&runtime, &fermion_u1(), "fZ2xU(1)", |case| {
            fermionic_blas_contract_oracle(case, role, twist)
        });
    }

    // Twisted control: `A: V⊗V ← V*⊗V*` with `B` (or `P'`) `V*⊗V* ← V⊗V`
    // puts dual legs on the core-right contracted codomain of the sorted
    // candidate. TensorKit copies an operand to twist it, and the result
    // must match TensorKit's B-role twist, not the untwisted sequence.
    let v = fermion_u1();
    let v_dual = v.try_dual().unwrap();
    let a = TensorMap::<_, D>::from_subblock_fn(&runtime, [&v, &v], [&v_dual, &v_dual], fill(4))
        .unwrap();
    let b = TensorMap::<_, D>::from_subblock_fn(&runtime, [&v_dual, &v_dual], [&v, &v], fill(5))
        .unwrap();
    let p = TensorMap::<_, D>::from_subblock_fn(&runtime, [&v, &v], [&v_dual, &v_dual], fill(6))
        .unwrap();
    for (name, rhs) in [("twisted C1", b), ("twisted L7", p.adjoint().unwrap())] {
        let case = Case {
            name,
            lhs: a.clone(),
            rhs,
            lhs_axes: vec![3, 2],
            rhs_axes: vec![1, 0],
            output_axes: vec![0, 1, 2, 3],
            dense: false,
        };
        let host = case.host();
        let twisted = fermionic_blas_contract_oracle(&case, TwistRole::B, twist);
        numerics::assert_nonzero_slices_close(
            name,
            host.dense_data().unwrap(),
            twisted.dense_data().unwrap(),
            case.terms(),
        );
        let untwisted = fermionic_blas_contract_oracle(&case, TwistRole::None, twist);
        assert!(
            host.dense_data()
                .unwrap()
                .iter()
                .zip(untwisted.dense_data().unwrap())
                .any(|(&x, &y)| x.distance(y) > 1e-3),
            "{name}: the twist changes nothing here"
        );
    }
}

#[test]
fn zero_copy_fermionic_candidates_match_tensorkit_with_and_without_twist() {
    let _serial = crate::counting_alloc::serial();
    // Serial: sibling tests read process-global transformer lookup counts.
    fermionic_values::<f64>();
    fermionic_values::<Complex64>();
    fermionic_values::<f32>();
    fermionic_values::<Complex32>();
}

/// Zero-copy candidates over non-square operands with a different space per
/// leg (`w*` is the dual of `w`):
///
/// - `N1`: `A: v⊗w ← w⊗v`, `A[3,2]·B[1,0]` with `B: w⊗v ← v` (sort);
/// - `D1`: the same with dual legs, `A: v⊗w* ← w*⊗v*`, `B: w*⊗v* ← w`;
/// - `S1`: `P'[0,1]·B[2,3] → [2,3,0,1]` with the lazy adjoint on the lhs,
///   `P: w⊗v ← v⊗w*` and `B: w⊗v ← v⊗w*` (swap);
/// - `S2`: rank 3 × 3 with one contracted leg, `A[0]·B[2] → [2,3,0,1]` with
///   `A: w ← v⊗w*` and `B: v⊗w ← w` (swap).
///
/// `N1` and `D1` contract only A's domain against B's codomain, so the
/// physical-basis expansion also applies to them.
fn mixed_cases<R, D>(runtime: &Runtime, v: &GradedSpace<R>, w: &GradedSpace<R>) -> Vec<Case<R, D>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let w_dual = w.try_dual().unwrap();
    let v_dual = v.try_dual().unwrap();
    let tensor = |codomain: &[&GradedSpace<R>], domain: &[&GradedSpace<R>], salt| {
        TensorMap::<R, D>::from_subblock_fn(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            fill(salt),
        )
        .unwrap()
    };
    let case = |name, lhs, rhs, l: &[usize], r: &[usize], out: &[usize], dense| Case {
        name,
        lhs,
        rhs,
        lhs_axes: l.to_vec(),
        rhs_axes: r.to_vec(),
        output_axes: out.to_vec(),
        dense,
    };
    vec![
        case(
            "N1",
            tensor(&[v, w], &[w, v], 71),
            tensor(&[w, v], &[v], 72),
            &[3, 2],
            &[1, 0],
            &[0, 1, 2],
            true,
        ),
        case(
            "D1",
            tensor(&[v, &w_dual], &[&w_dual, &v_dual], 73),
            tensor(&[&w_dual, &v_dual], &[w], 74),
            &[3, 2],
            &[1, 0],
            &[0, 1, 2],
            true,
        ),
        case(
            "S1",
            tensor(&[w, v], &[v, &w_dual], 75).adjoint().unwrap(),
            tensor(&[w, v], &[v, &w_dual], 76),
            &[0, 1],
            &[2, 3],
            &[2, 3, 0, 1],
            false,
        ),
        case(
            "S2",
            tensor(&[w], &[v, &w_dual], 77),
            tensor(&[v, w], &[w], 78),
            &[0],
            &[2],
            &[2, 3, 0, 1],
            false,
        ),
    ]
}

fn mixed_values<R, D>(v: &GradedSpace<R>, w: &GradedSpace<R>, symmetry: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + symtn::sector::PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    let runtime = Runtime::builder().build().unwrap();
    for case in mixed_cases::<R, D>(&runtime, v, w) {
        let what = format!("{symmetry} {} [{}]", case.name, D::NAME);
        let host = case.host();
        numerics::assert_nonzero_slices_close(
            &what,
            host.dense_data().unwrap(),
            blas_contract_oracle(&case).dense_data().unwrap(),
            case.terms(),
        );
        if case.dense {
            let (shape, expected) = dense_oracle(&case);
            let actual = host.to_physical_dense().unwrap();
            assert_eq!(actual.shape, shape, "{what}");
            numerics::assert_nonzero_slices_close(&what, &actual.data, &expected, case.terms());
        }
    }
}

fn assert_mixed_zero_copy<R>(v: &GradedSpace<R>, w: &GradedSpace<R>, symmetry: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for case in mixed_cases::<R, f64>(&runtime, v, w) {
        let (_, _, lookups) = warm(&runtime, &case);
        // What: the selected candidate transforms neither source nor output.
        assert_eq!(lookups, 0, "{symmetry} {}: tree transforms ran", case.name);
    }
}

fn u1_second() -> GradedSpace<symtn::sector::U1FusionRule> {
    u1(&[(0, 2), (1, 1), (2, 1)])
}

fn su2_second() -> GradedSpace<symtn::sector::SU2FusionRule> {
    GradedSpace::try_new(
        std::sync::Arc::new(symtn::sector::SU2FusionRule),
        [
            (symtn::sector::SU2Irrep::from_twice_spin(1), 1),
            (symtn::sector::SU2Irrep::from_twice_spin(2), 2),
        ],
    )
    .unwrap()
}

#[test]
fn mixed_space_candidates_match_the_blas_sequence_and_the_dense_expansion() {
    let _serial = crate::counting_alloc::serial();
    // Serial: sibling tests read process-global transformer lookup counts.
    fn at<D: Payload>() {
        mixed_values::<_, D>(&u1_non_self_dual(), &u1_second(), "U(1)");
        mixed_values::<_, D>(&su2(), &su2_second(), "SU(2)");
    }
    at::<f64>();
    at::<Complex64>();
    at::<f32>();
    at::<Complex32>();
}

#[test]
fn mixed_space_candidates_run_no_transform() {
    let _serial = crate::counting_alloc::serial();
    // Its helper serializes: sibling tests read process-global lookup counts.
    assert_mixed_zero_copy(&u1_non_self_dual(), &u1_second(), "U(1)");
    assert_mixed_zero_copy(&su2(), &su2_second(), "SU(2)");
}

fn transform_lookups(_: &Runtime) -> usize {
    let info = completed_transformers();
    (info.hits() + info.misses()) as usize
}

/// The zero-copy candidate with its own output order, then one `permute`.
fn contract_then_permute<R, D>(case: &Case<R, D>, swapped: bool) -> TensorMap<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let (temporary, codomain, domain) = copy_c_steps(case, swapped);
    temporary.permute(&codomain, &domain).unwrap()
}

/// TensorKit `copyC`'s two steps spelled with public operations: the
/// zero-copy candidate contraction with its own output order, and the
/// permutation (codomain, domain) that places it in the requested order.
fn copy_c_steps<R, D>(case: &Case<R, D>, swapped: bool) -> (TensorMap<R, D>, Vec<usize>, Vec<usize>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let lhs_open = case.lhs.rank() - case.lhs_axes.len();
    let rhs_open = case.rhs.rank() - case.rhs_axes.len();
    let (temporary, position): (_, Box<dyn Fn(usize) -> usize>) = if swapped {
        let identity: Vec<usize> = (0..lhs_open + rhs_open).collect();
        (
            case.rhs
                .contract(
                    &case.lhs,
                    &ContractSpec {
                        lhs: &case.rhs_axes,
                        rhs: &case.lhs_axes,
                        codomain: &identity[..rhs_open],
                        domain: &identity[rhs_open..],
                    },
                )
                .unwrap(),
            Box::new(move |axis| {
                if axis < lhs_open {
                    axis + rhs_open
                } else {
                    axis - lhs_open
                }
            }),
        )
    } else {
        let identity: Vec<usize> = (0..lhs_open + rhs_open).collect();
        (
            case.lhs
                .contract(
                    &case.rhs,
                    &ContractSpec {
                        lhs: &case.lhs_axes,
                        rhs: &case.rhs_axes,
                        codomain: &identity[..lhs_open],
                        domain: &identity[lhs_open..],
                    },
                )
                .unwrap(),
            Box::new(|axis| axis),
        )
    };
    let mut codomain: Vec<usize> = case
        .output_axes
        .iter()
        .map(|&axis| position(axis))
        .collect();
    let domain = codomain.split_off(lhs_open);
    (temporary, codomain, domain)
}

fn assert_output_permute_budget<R>(v: &GradedSpace<R>, symmetry: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (case, swapped) in copy_c_probes::<R, f64>(&runtime, v) {
        let (calls, bytes, lookups) = warm(&runtime, &case);
        contract_then_permute(&case, swapped);
        let before = transform_lookups(&runtime);
        let (two_step, allocs) =
            counting_alloc::measure(|| contract_then_permute(black_box(&case), swapped));
        drop(two_step);
        let (budget_calls, budget_bytes) = (allocs.calls, allocs.bytes);
        let permute_lookups = transform_lookups(&runtime) - before;
        let name = case.name;
        eprintln!(
            "{symmetry} {name}: {calls} calls, {bytes} B, {lookups} transform lookups; \
             contract + permute {budget_calls} / {budget_bytes} / {permute_lookups}"
        );
        // What: only the output permute transforms; no source is rebuilt. The
        // planned route looks its one transform up once (the facade `permute`
        // of the two-step budget takes two lookups for the same structure).
        assert!(
            lookups == 1 && lookups <= permute_lookups,
            "{symmetry} {name}: source transforms ran"
        );
        // What: no more than the zero-copy contract plus one permute, so no
        // operand (a `P'`-sized buffer) is materialized.
        assert!(
            calls <= budget_calls && bytes <= budget_bytes,
            "{symmetry} {name}: {calls} calls / {bytes} B vs contract + permute \
             {budget_calls} / {budget_bytes}"
        );
    }
}

#[test]
fn zero_copy_candidates_with_an_output_permute_allocate_like_contract_then_permute() {
    let _serial = crate::counting_alloc::serial();
    // Its helper serializes: sibling tests read process-global lookup counts.
    assert_output_permute_budget(&u1_non_self_dual(), "U(1)");
    assert_output_permute_budget(&su2(), "SU(2)");
    assert_output_permute_budget(&fermion_u1(), "fZ2xU(1)");
}

/// `S3`: rank 3 × 4 with open counts 2 and 3, so the swapped candidate's own
/// output has a different codomain rank from the result's, and operands on
/// distinct provider allocations: `P'[0]·B[3] → [2,0,1,3,4]` with
/// `P: v⊗w* ← w` and `B: w⊗v⊗v ← w`.
fn uneven_swap<R, D>(runtime: &Runtime, v: &GradedSpace<R>, w: &GradedSpace<R>) -> Case<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let w_dual = w.try_dual().unwrap();
    Case {
        name: "S3",
        lhs: TensorMap::from_subblock_fn(runtime, [v, &w_dual], [w], fill(84))
            .unwrap()
            .adjoint()
            .unwrap(),
        rhs: TensorMap::from_subblock_fn(runtime, [w, v, v], [w], fill(85)).unwrap(),
        lhs_axes: vec![0],
        rhs_axes: vec![3],
        output_axes: vec![2, 0, 1, 3, 4],
        dense: false,
    }
}

fn output_permute_values<R, D>(v: &GradedSpace<R>, w: &GradedSpace<R>, symmetry: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + symtn::sector::PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    let runtime = Runtime::builder().build().unwrap();
    let cases = copy_c_probes::<R, D>(&runtime, v)
        .into_iter()
        .map(|(case, _)| case)
        .chain([uneven_swap::<R, D>(&runtime, v, w)]);
    for case in cases {
        let what = format!("{symmetry} {} [{}]", case.name, D::NAME);
        let host = case.host();
        // What: the left-authority provider rule holds after a swap.
        assert!(std::ptr::eq(host.provider(), case.lhs.provider()), "{what}");
        assert_eq!(host.codomain_rank(), case.lhs.rank() - case.lhs_axes.len());
        numerics::assert_nonzero_slices_close(
            &what,
            host.dense_data().unwrap(),
            blas_contract_oracle(&case).dense_data().unwrap(),
            case.terms(),
        );
        if case.dense {
            let (shape, expected) = dense_oracle(&case);
            let actual = host.to_physical_dense().unwrap();
            assert_eq!(actual.shape, shape, "{what}");
            numerics::assert_nonzero_slices_close(&what, &actual.data, &expected, case.terms());
        }
    }
}

#[test]
fn zero_copy_candidates_with_an_output_permute_match_tensorkit_and_the_dense_expansion() {
    let _serial = crate::counting_alloc::serial();
    // Serial: sibling tests read process-global transformer lookup counts.
    fn at<D: Payload>() {
        output_permute_values::<_, D>(&u1_non_self_dual(), &u1_second(), "U(1)");
        output_permute_values::<_, D>(&su2(), &su2_second(), "SU(2)");
    }
    at::<f64>();
    at::<Complex64>();
    at::<f32>();
    at::<Complex32>();
}

/// #1626: the pooled copyC temporary changes no arithmetic. Every probe
/// equals its literal zero-copy contraction then permute, bit for bit.
fn output_permute_bits<R, D>(v: &GradedSpace<R>, w: &GradedSpace<R>, symmetry: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload + PartialEq,
{
    let runtime = Runtime::builder().build().unwrap();
    let cases = copy_c_probes::<R, D>(&runtime, v)
        .into_iter()
        .chain([(uneven_swap::<R, D>(&runtime, v, w), true)]);
    for (case, swapped) in cases {
        let what = format!("{symmetry} {} [{}]", case.name, D::NAME);
        let expected = contract_then_permute(&case, swapped);
        for _ in 0..2 {
            let actual = case.host();
            assert_eq!(actual.codomain(), expected.codomain(), "{what}");
            assert_eq!(actual.domain(), expected.domain(), "{what}");
            assert!(
                actual.dense_data().unwrap() == expected.dense_data().unwrap(),
                "{what}"
            );
        }
    }
}

/// #1631: `contract_into` takes the same copyC route as `contract`, so it
/// equals the two copyC steps with the caller's α/β on the permute
/// (`tensoradd!(C, Cnew, pAB, false, α, β)`), bit for bit, for β = 0 (over a
/// NaN destination), β = 1 and a general β, with owned and lazy operands.
/// A warm call looks up one transform, the output permute, and no more than
/// a warm `contract` does.
fn into_copy_c_bits<R, D>(v: &GradedSpace<R>, w: &GradedSpace<R>, symmetry: &str)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload + PartialEq,
{
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let cases = copy_c_probes::<R, D>(&runtime, v)
        .into_iter()
        .chain([(uneven_swap::<R, D>(&runtime, v, w), true)]);
    let alpha = D::entry(1.25, -0.5);
    for (case, swapped) in cases {
        let (temporary, codomain, domain) = copy_c_steps(&case, swapped);
        let result = case.host();
        for (beta, seed) in [
            (D::entry(0.0, 0.0), D::entry(f64::NAN, 0.0)),
            (D::entry(1.0, 0.0), D::entry(0.75, 0.125)),
            (D::entry(-0.625, 0.25), D::entry(0.75, 0.125)),
        ] {
            let what = format!("{symmetry} {} [{}] beta {beta:?}", case.name, D::NAME);
            let mut expected = result.scale(seed).unwrap();
            temporary
                .permute_into(&codomain, &domain, &mut expected, alpha, beta)
                .unwrap();
            let mut actual = result.scale(seed).unwrap();
            case.lhs
                .contract_into(&case.rhs, &case.spec(), &mut actual, alpha, beta)
                .unwrap();
            assert!(
                actual.dense_data().unwrap() == expected.dense_data().unwrap(),
                "{what}"
            );
        }

        let mut destination = result.scale(D::entry(1.0, 0.0)).unwrap();
        let into = |destination: &mut TensorMap<R, D>| {
            case.lhs
                .contract_into(
                    &case.rhs,
                    &case.spec(),
                    destination,
                    D::entry(1.0, 0.0),
                    D::entry(0.0, 0.0),
                )
                .unwrap();
        };
        into(&mut destination);
        into(&mut destination);
        let before = transform_lookups(&runtime);
        into(&mut destination);
        let into_lookups = transform_lookups(&runtime) - before;
        drop(case.host());
        let before = transform_lookups(&runtime);
        drop(case.host());
        let contract_lookups = transform_lookups(&runtime) - before;
        // What: the output permute is the only transform (no operand is
        // rebuilt), and never more lookups than the owned route, whose owned
        // overwrite probe may add one.
        assert!(
            into_lookups == 1 && into_lookups <= contract_lookups,
            "{symmetry} {} [{}]: warm contract_into {into_lookups} vs contract \
             {contract_lookups} lookups",
            case.name,
            D::NAME
        );
    }
}

#[test]
fn contract_into_takes_the_copy_c_route_of_contract() {
    let _serial = crate::counting_alloc::serial();
    // Its helper serializes: sibling tests read process-global lookup counts.
    into_copy_c_bits::<_, f64>(&u1_non_self_dual(), &u1_second(), "U(1)");
    into_copy_c_bits::<_, Complex64>(&u1_non_self_dual(), &u1_second(), "U(1)");
    into_copy_c_bits::<_, f64>(&su2(), &su2_second(), "SU(2)");
    into_copy_c_bits::<_, Complex64>(&su2(), &su2_second(), "SU(2)");
}

#[test]
fn zero_copy_candidates_with_an_output_permute_equal_contract_then_permute_bitwise() {
    let _serial = crate::counting_alloc::serial();
    // Serial: sibling tests read process-global transformer lookup counts.
    output_permute_bits::<_, f64>(&u1_non_self_dual(), &u1_second(), "U(1)");
    output_permute_bits::<_, Complex64>(&su2(), &su2_second(), "SU(2)");
}

#[test]
fn uneven_swapped_candidate_with_an_output_permute_runs_one_transform() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let case = uneven_swap::<_, f64>(&runtime, &u1_non_self_dual(), &u1_second());
    let (_, _, lookups) = warm(&runtime, &case);
    contract_then_permute(&case, true);
    let before = transform_lookups(&runtime);
    contract_then_permute(&case, true);
    // What: only the output permute transforms: one lookup of its one
    // structure (the facade `permute` takes two lookups for it).
    assert!(
        lookups == 1 && lookups <= transform_lookups(&runtime) - before,
        "S3: source transforms ran"
    );
}

fn fermionic_output_permute_values<D: Payload>() {
    let runtime = Runtime::builder().build().unwrap();
    let twist =
        |t: &TensorMap<FermionU1, D>, legs: &[usize]| t.twist(legs, Direction::Forward).unwrap();
    // As for the identity-output probes, the literal sequence of the swapped
    // probes twists dual `B` legs that the selected candidate does not, so
    // either twist role is the oracle.
    for role in [TwistRole::B, TwistRole::A] {
        for (case, _) in copy_c_probes::<_, D>(&runtime, &fermion_u1()) {
            let expected = fermionic_blas_contract_oracle(&case, role, twist);
            numerics::assert_nonzero_slices_close(
                &format!("fZ2xU(1) {} [{}]", case.name, D::NAME),
                case.host().dense_data().unwrap(),
                expected.dense_data().unwrap(),
                case.terms(),
            );
        }
    }

    // Twisted control: dual legs on the core-right contracted codomain of the
    // sorted candidate, under a permuted output. The candidate is declined,
    // and the result must carry TensorKit's B-role twist.
    let v = fermion_u1();
    let v_dual = v.try_dual().unwrap();
    let a = TensorMap::<_, D>::from_subblock_fn(&runtime, [&v, &v], [&v_dual, &v_dual], fill(86))
        .unwrap();
    let p = TensorMap::<_, D>::from_subblock_fn(&runtime, [&v, &v], [&v_dual, &v_dual], fill(87))
        .unwrap();
    let case = Case {
        name: "twisted L7p",
        lhs: a,
        rhs: p.adjoint().unwrap(),
        lhs_axes: vec![3, 2],
        rhs_axes: vec![1, 0],
        output_axes: vec![1, 0, 2, 3],
        dense: false,
    };
    let expected = fermionic_blas_contract_oracle(&case, TwistRole::B, twist);
    numerics::assert_nonzero_slices_close(
        case.name,
        case.host().dense_data().unwrap(),
        expected.dense_data().unwrap(),
        case.terms(),
    );
}

#[test]
fn zero_copy_fermionic_candidates_with_an_output_permute_match_tensorkit() {
    let _serial = crate::counting_alloc::serial();
    // Serial: sibling tests read process-global transformer lookup counts.
    fermionic_output_permute_values::<f64>();
    fermionic_output_permute_values::<Complex64>();
    fermionic_output_permute_values::<f32>();
    fermionic_output_permute_values::<Complex32>();
}

/// Review inputs of #1475 where `dim(C)` dwarfs the operands, so
/// TensorKit's `_contract_memcost` (tensoroperations.jl L378) prefers
/// copying both operands (m3/m4, `dim(A) + dim(B)`) over `copyC`: with
/// `v = u1{-1:16, 0:16, 1:16}` and `c = u1{0:1}`,
///
/// - `R1` `A[3,2]·Q'[1,0] → [2,3,0,1]` with `A: v⊗v ← c⊗c`,
///   `Q = v⊗v ← c⊗c`;
/// - `R2` `P'[3,2]·B[1,0] → [2,3,0,1]` with `P: c⊗c ← v⊗v`,
///   `B: c⊗c ← v⊗v`;
/// - `R3` the owned sort `A[3,2]·B[1,0] → [2,3,0,1]`;
/// - `R4` the #1466 core form `A[2,3]·B[0,1] → [2,3,0,1]`.
fn large_output_cases(runtime: &Runtime) -> Vec<Case<symtn::sector::U1FusionRule, f64>> {
    let v = u1(&[(-1, 16), (0, 16), (1, 16)]);
    let c = u1(&[(0, 1)]);
    let tensor = |codomain: [&GradedSpace<_>; 2], domain: [&GradedSpace<_>; 2], salt| {
        TensorMap::<_, f64>::from_subblock_fn(runtime, codomain, domain, fill(salt)).unwrap()
    };
    let a = tensor([&v, &v], [&c, &c], 91);
    let q = tensor([&v, &v], [&c, &c], 92).adjoint().unwrap();
    let p = tensor([&c, &c], [&v, &v], 93).adjoint().unwrap();
    let b = tensor([&c, &c], [&v, &v], 94);
    let case =
        |name, lhs: &TensorMap<_, f64>, rhs: &TensorMap<_, f64>, l: [usize; 2], r: [usize; 2]| {
            Case {
                name,
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                lhs_axes: l.to_vec(),
                rhs_axes: r.to_vec(),
                output_axes: vec![2, 3, 0, 1],
                dense: false,
            }
        };
    vec![
        case("R1", &a, &q, [3, 2], [1, 0]),
        case("R2", &p, &b, [3, 2], [1, 0]),
        case("R3", &a, &b, [3, 2], [1, 0]),
        case("R4", &a, &b, [2, 3], [0, 1]),
    ]
}

/// Review inputs where the destination has coupled sectors no GEMM writes,
/// so `dim(C)` (946,176) is far above its GEMM-active part (200,704) and
/// above `dim(A) + dim(B)`: with `v = u1{-3..3, each 8}` and `w = u1{0:20}`,
///
/// - `X1` `A[3,2]·B[1,0] → [2,3,0,1]`, `A: v⊗v ← w⊗w`, `B: w⊗w ← v⊗v`;
/// - `X2` `A[3,2]·Q'[1,0]`, `Q: v⊗v ← w⊗w`;
/// - `X3` `P'[3,2]·B[1,0]`, `P: w⊗w ← v⊗v`;
/// - `X4` the #1466 core form `A[2]·B[0] → [2,3,0,1]` with
///   `A: v⊗v ← u`, `B: u ← v⊗v`, `u = u1{0:500}`.
fn inactive_output_cases(runtime: &Runtime) -> Vec<Case<symtn::sector::U1FusionRule, f64>> {
    let v = u1(&[(-3, 8), (-2, 8), (-1, 8), (0, 8), (1, 8), (2, 8), (3, 8)]);
    let w = u1(&[(0, 20)]);
    let u = u1(&[(0, 500)]);
    let tensor = |codomain: &[&GradedSpace<_>], domain: &[&GradedSpace<_>], salt| {
        TensorMap::<_, f64>::from_subblock_fn(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            fill(salt),
        )
        .unwrap()
    };
    let a = tensor(&[&v, &v], &[&w, &w], 95);
    let b = tensor(&[&w, &w], &[&v, &v], 96);
    let q = tensor(&[&v, &v], &[&w, &w], 97).adjoint().unwrap();
    let p = tensor(&[&w, &w], &[&v, &v], 98).adjoint().unwrap();
    let case =
        |name, lhs: &TensorMap<_, f64>, rhs: &TensorMap<_, f64>, l: &[usize], r: &[usize]| Case {
            name,
            lhs: lhs.clone(),
            rhs: rhs.clone(),
            lhs_axes: l.to_vec(),
            rhs_axes: r.to_vec(),
            output_axes: vec![2, 3, 0, 1],
            dense: false,
        };
    vec![
        case("X1", &a, &b, &[3, 2], &[1, 0]),
        case("X2", &a, &q, &[3, 2], &[1, 0]),
        case("X3", &p, &b, &[3, 2], &[1, 0]),
        case(
            "X4",
            &tensor(&[&v, &v], &[&u], 99),
            &tensor(&[&u], &[&v, &v], 100),
            &[2],
            &[0],
        ),
    ]
}

#[test]
fn large_output_from_small_operands_copies_the_operands_not_c() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for case in large_output_cases(&runtime)
        .into_iter()
        .chain(inactive_output_cases(&runtime))
    {
        let name = case.name;
        let host = case.host();
        let output_bytes = std::mem::size_of_val(host.dense_data().unwrap()) as u64;
        numerics::assert_nonzero_slices_close(
            name,
            host.dense_data().unwrap(),
            blas_contract_oracle(&case).dense_data().unwrap(),
            case.terms(),
        );
        let (calls, bytes, lookups) = warm(&runtime, &case);
        eprintln!("U(1) {name}: {calls} calls, {bytes} B, {lookups} transform lookups, dim(C) {output_bytes} B");
        // What: only the output is allocated, not a `dim(C)` temporary as well.
        assert!(
            bytes < output_bytes + output_bytes / 2,
            "{name}: {bytes} B allocated for a {output_bytes} B output"
        );
    }
}

/// Warm DynamicTree calls on the inactive-destination inputs, each after the
/// previous result was dropped, as in a sweep: the destination's coupled
/// regions are then recompiled on a fresh structure wrapper every call, so
/// their per-sector cost is visible (#1488). The regions cost one allocation
/// per coupled sector and a run of inactive sectors one scale layout, not a
/// per-tree cost (191 calls for X1 and X4 at d90bfd17).
#[test]
fn dynamic_tree_with_a_fresh_destination_allocates_per_region_not_per_tree() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for case in inactive_output_cases(&runtime) {
        let (calls, bytes, _) = warm(&runtime, &case);
        let name = case.name;
        eprintln!("U(1) {name}: {calls} calls, {bytes} B with a fresh destination");
        // What: 13 destination sectors do not cost ~10 allocations each.
        assert!(calls <= 70, "{name}: {calls} warm allocation calls");
    }
}

/// `dim(C)` below both operands, so `copyC` wins TensorKit's
/// `_contract_memcost` without scoring the DynamicTree candidates (#1488):
/// with `v = u1{-1:8, 0:8, 1:8}` and `c = u1{0:2, 1:1}`,
///
/// - `S1` the sort `A[3,2]·B[1,0] → [2,3,0,1]`, `A: c⊗c ← v⊗v`,
///   `B: v⊗v ← c⊗c`;
/// - `S2` the swap `A[0,1]·B[2,3] → [3,2,1,0]`, `A: v⊗v ← c⊗c`,
///   `B: c⊗c ← v⊗v`.
///
/// Values against TensorKit's `blas_contract!` sequence, and one warm tree
/// transform: the output permute, with neither operand rebuilt. The route
/// itself (copyC, operands read untransformed) is pinned on these shapes by
/// symtn-tensors' `small_output_takes_copy_c_without_copying_an_operand`;
/// that the shortcut skips the DynamicTree scorer is established by source
/// inspection of `copy_c_order` (debug builds compare it with full scoring),
/// not asserted here. The CopyC allocation budget is
/// `contract_route_host_allocations`'s contract.
#[test]
fn small_output_copy_c_shapes_match_tensorkit_with_one_output_transform() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = u1(&[(-1, 8), (0, 8), (1, 8)]);
    let c = u1(&[(0, 2), (1, 1)]);
    let tensor = |codomain: [&GradedSpace<_>; 2], domain: [&GradedSpace<_>; 2], salt| {
        TensorMap::<_, f64>::from_subblock_fn(&runtime, codomain, domain, fill(salt)).unwrap()
    };
    let case = |name, lhs, rhs, l: [usize; 2], r: [usize; 2], out: [usize; 4]| Case {
        name,
        lhs,
        rhs,
        lhs_axes: l.to_vec(),
        rhs_axes: r.to_vec(),
        output_axes: out.to_vec(),
        dense: false,
    };
    let cases = [
        case(
            "S1",
            tensor([&c, &c], [&v, &v], 101),
            tensor([&v, &v], [&c, &c], 102),
            [3, 2],
            [1, 0],
            [2, 3, 0, 1],
        ),
        case(
            "S2",
            tensor([&v, &v], [&c, &c], 103),
            tensor([&c, &c], [&v, &v], 104),
            [0, 1],
            [2, 3],
            [3, 2, 1, 0],
        ),
    ];
    for case in &cases {
        let name = case.name;
        numerics::assert_nonzero_slices_close(
            name,
            case.host().dense_data().unwrap(),
            blas_contract_oracle(case).dense_data().unwrap(),
            case.terms(),
        );
        let (calls, bytes, lookups) = warm(&runtime, case);
        eprintln!("U(1) {name}: {calls} calls, {bytes} B, {lookups} transform lookups");
        // What: the output permute is the only tree transform.
        assert_eq!(lookups, 1, "{name}: transform lookups");
    }
}

/// The one shape the `copyC` shortcut still scores (#1488): a swap whose
/// `dim(C)` ties the smaller operand and whose requested order is the
/// identity, the only order in which an A·B candidate copies no output:
/// `A[0,1]·B[2,3] → [0,1,2,3]` over `v⊗v ← v⊗v`. Debug builds check the
/// decision against full scoring; the value is TensorKit's.
#[test]
fn swapped_tie_with_an_identity_output_matches_tensorkit() {
    let _serial = crate::counting_alloc::serial();
    // Serial: sibling tests read process-global transformer lookup counts.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let v = u1_non_self_dual();
    let tensor =
        |salt| TensorMap::<_, f64>::from_subblock_fn(&runtime, [&v, &v], [&v, &v], fill(salt));
    let case = Case {
        name: "T1",
        lhs: tensor(105).unwrap(),
        rhs: tensor(106).unwrap(),
        lhs_axes: vec![0, 1],
        rhs_axes: vec![2, 3],
        output_axes: vec![0, 1, 2, 3],
        dense: false,
    };
    numerics::assert_nonzero_slices_close(
        case.name,
        case.host().dense_data().unwrap(),
        blas_contract_oracle(&case).dense_data().unwrap(),
        case.terms(),
    );
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
