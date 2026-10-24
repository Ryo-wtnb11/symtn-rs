//! One facade variant per axis-misuse kind (#1873): a malformed permutation
//! is `OperationError::InvalidPermutation { axes, rank }` and a malformed axis
//! subset is `OperationError::InvalidAxisSet { tensor, axes, rank }` at every
//! entry point — transform, contract, trace, factorization leg roles, eager
//! and into a destination — on Host and CUDA, in the multiplicity-free and
//! checked Generic modes, for `f64` and `Complex64`, on legs with several
//! sectors and degeneracies above one.
//!
//! The expected kind follows TensorKit's argument checks (an output-axis list
//! must be a permutation, `isperm`; contracted and traced axes are distinct
//! axes of the operand), not the code under test. The device variant runs
//! with `--features cuda -- --ignored`.

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{
    Complex64, ContractSpec, Error, GradedSpace, OperationError, Runtime, TensorMap,
};

fn permutation(error: &Error, axes: &[usize], rank: usize) {
    assert!(
        matches!(
            error,
            Error::Operation(operation)
                if **operation == OperationError::InvalidPermutation { axes: axes.to_vec(), rank }
        ),
        "expected InvalidPermutation {axes:?}/{rank}, got {error:?}"
    );
}

fn subset(error: &Error, tensor: &str, axes: &[usize], rank: usize) {
    assert!(
        matches!(
            error,
            Error::Operation(operation)
                if matches!(
                    &**operation,
                    OperationError::InvalidAxisSet { tensor: got, axes: got_axes, rank: got_rank }
                        if *got == tensor && got_axes == axes && *got_rank == rank
                )
        ),
        "expected InvalidAxisSet {tensor} {axes:?}/{rank}, got {error:?}"
    );
}

/// The malformed output-axis lists of a rank-4 tensor, as `(codomain,
/// domain)`: out of range, repeated, too short.
fn bad_permutations() -> [(Vec<usize>, Vec<usize>); 3] {
    [
        (vec![0, 9], vec![2, 3]),
        (vec![0, 0], vec![2, 3]),
        (vec![0], vec![2, 3]),
    ]
}

/// Every entry point, eager; `$facade` turns an entry's error into the
/// facade `Error` (identity for multiplicity-free, `Facade` for Generic).
macro_rules! eager_matrix {
    ($t:expr, $facade:expr) => {{
        let t = &$t;
        let facade = $facade;
        for (codomain, domain) in bad_permutations() {
            let linear: Vec<usize> = codomain.iter().chain(&domain).copied().collect();
            permutation(
                &facade(t.permute(&codomain, &domain).unwrap_err()),
                &linear,
                4,
            );
            permutation(
                &facade(t.transpose(&codomain, &domain).unwrap_err()),
                &linear,
                4,
            );
            permutation(
                &facade(t.braid(&codomain, &domain, &[0, 1, 2, 3]).unwrap_err()),
                &linear,
                4,
            );
            permutation(
                &facade(t.svd_compact(&codomain, &domain).unwrap_err()),
                &linear,
                4,
            );
            let spec = ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &codomain,
                domain: &domain,
            };
            permutation(&facade(t.contract(t, &spec).unwrap_err()), &linear, 4);
        }
        for axes in [vec![2, 9], vec![2, 2]] {
            let spec = ContractSpec {
                lhs: &axes,
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2, 3],
            };
            subset(&facade(t.contract(t, &spec).unwrap_err()), "lhs", &axes, 4);
            let spec = ContractSpec {
                lhs: &[2, 3],
                rhs: &axes,
                codomain: &[0, 1],
                domain: &[2, 3],
            };
            subset(&facade(t.contract(t, &spec).unwrap_err()), "rhs", &axes, 4);
        }
        subset(
            &facade(t.trace_pairs(&[(0, 9)]).unwrap_err()),
            "trace pairs",
            &[0, 9],
            4,
        );
        subset(
            &facade(t.trace_pairs(&[(0, 0)]).unwrap_err()),
            "trace pairs",
            &[0, 0],
            4,
        );
    }};
}

/// The multiplicity-free `_into` entries, given a placement `$lift` and a
/// destination of the source's own space.
macro_rules! into_matrix {
    ($t:expr, $fresh:expr, $one:expr) => {{
        let t = &$t;
        let fresh = $fresh;
        let (one, zero) = ($one, $one - $one);
        for (codomain, domain) in bad_permutations() {
            let linear: Vec<usize> = codomain.iter().chain(&domain).copied().collect();
            permutation(
                &t.permute_into(&codomain, &domain, &mut fresh(), one, zero)
                    .unwrap_err(),
                &linear,
                4,
            );
            permutation(
                &t.transpose_into(&codomain, &domain, &mut fresh(), one, zero)
                    .unwrap_err(),
                &linear,
                4,
            );
            permutation(
                &t.braid_into(&codomain, &domain, &[0, 1, 2, 3], &mut fresh(), one, zero)
                    .unwrap_err(),
                &linear,
                4,
            );
            let spec = ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &codomain,
                domain: &domain,
            };
            permutation(
                &t.contract_into(t, &spec, &mut fresh(), one, zero)
                    .unwrap_err(),
                &linear,
                4,
            );
        }
        for axes in [vec![2, 9], vec![2, 2]] {
            let spec = ContractSpec {
                lhs: &axes,
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2, 3],
            };
            subset(
                &t.contract_into(t, &spec, &mut fresh(), one, zero)
                    .unwrap_err(),
                "lhs",
                &axes,
                4,
            );
        }
        subset(
            &t.trace_pairs_into(&[(0, 9)], &mut fresh(), one, zero)
                .unwrap_err(),
            "trace pairs",
            &[0, 9],
            4,
        );
    }};
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    let j = |twice: usize| SU2Irrep::from_twice_spin(twice);
    GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 1), (j(2), 2)]).unwrap()
}

macro_rules! multiplicity_free {
    ($rt:expr, $d:ty, $lift:expr) => {{
        let rt = $rt;
        let lift = $lift;
        let v = su2_leg();
        let vd = v.try_dual().unwrap();
        let host = TensorMap::<_, $d>::rand_with_seed(rt, [&v, &vd], [&v, &vd], 3).unwrap();
        let t = lift(&host);
        eager_matrix!(t, |error: Error| error);
        into_matrix!(t, || lift(&host.zeros_like().unwrap()), <$d>::from(1.0));
    }};
}

#[test]
fn multiplicity_free_host_axis_errors_have_one_variant_per_kind() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    multiplicity_free!(&rt, f64, |x: &TensorMap<_, f64>| x.clone());
    multiplicity_free!(&rt, Complex64, |x: &TensorMap<_, Complex64>| x.clone());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn multiplicity_free_device_axis_errors_match_the_host() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    multiplicity_free!(&rt, f64, |x: &TensorMap<_, f64>| x.to_cuda().unwrap());
    multiplicity_free!(&rt, Complex64, |x: &TensorMap<_, Complex64>| x
        .to_cuda()
        .unwrap());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_axis_errors_are_the_multiplicity_free_facade_errors() {
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::GenericTensorError;

    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![0i64, 0], 2), (vec![1, 0], 1), (vec![1, 1], 2)],
    )
    .unwrap();
    let vd = v.try_dual().unwrap();
    fn facade<E: std::fmt::Debug>(error: GenericTensorError<E>) -> Error {
        match error {
            GenericTensorError::Facade(error) => error,
            other => panic!("axis misuse must be a facade error, got {other:?}"),
        }
    }
    let t = TensorMap::<_, f64>::rand_with_seed(&rt, [&v, &vd], [&v, &vd], 3).unwrap();
    eager_matrix!(t, facade);
    let t = TensorMap::<_, Complex64>::rand_with_seed(&rt, [&v, &vd], [&v, &vd], 3).unwrap();
    eager_matrix!(t, facade);
}
