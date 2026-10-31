//! `DenseTreeTransformOperations` runs the one batched-GEMM replay.
//! Its values are checked against the explicit-index oracle in `common`
//! (`Fixture::expected_scaled`), which walks each fixture's block metadata and
//! never reads the compiled structure; the per-block strided driver serves only
//! as the bitwise dispatch witness.

use crate::common;

use common::{all_fixtures, recoupling_fixtures, Fixture, TestScalar};
use num_complex::Complex64;
use symtn_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw,
    tree_transform_structure_with_strided_kernel_raw, ConjugateValue, DenseRecouplingScalar,
    DenseTreeTransformOperations, RecouplingCoefficientAction, StridedHostKernelAdapter,
    TreeTransformBackend, TreeTransformScalar, TreeTransformWorkspace,
};

struct Replays<T> {
    dense: Vec<T>,
    reference: Vec<T>,
}

/// Non-dyadic payload so the GEMM and scalar-loop summation orders round
/// differently.
fn payload<T: TestScalar>(fixture: &Fixture) -> Vec<T> {
    (0..fixture.src_len())
        .map(|i| T::from_parts((i as f64 * 0.731 + 0.3).sin(), (i as f64 * 1.37).cos()))
        .collect()
}

fn initial<T: TestScalar>(fixture: &Fixture) -> Vec<T> {
    (0..fixture.dst_len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect()
}

fn replay<T>(fixture: &Fixture, overwrite: bool, alpha: T) -> Replays<T>
where
    T: TestScalar
        + TreeTransformScalar
        + RecouplingCoefficientAction<f64>
        + DenseRecouplingScalar
        + ConjugateValue,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let structure = fixture.compile();
    let dst_structure = fixture.dst_structure();
    let src_structure = fixture.src_structure();
    let source = payload::<T>(fixture);
    let initial = initial::<T>(fixture);
    let beta = T::from_parts(1.0, 0.0);

    let mut dense = initial.clone();
    let mut reference = initial;
    let mut dense_backend = DenseTreeTransformOperations::default();
    if overwrite {
        dense_backend
            .tree_transform_structure_overwrite_into_raw(
                engine,
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dense,
                &source,
                alpha,
                &[],
            )
            .unwrap();
        tree_transform_structure_overwrite_with_strided_kernel_raw(
            strided_kernel::Exec::serial(),
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut reference,
            &source,
            alpha,
        )
        .unwrap();
    } else {
        dense_backend
            .tree_transform_structure_into_raw(
                engine,
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dense,
                &source,
                alpha,
                beta,
            )
            .unwrap();
        tree_transform_structure_with_strided_kernel_raw(
            strided_kernel::Exec::serial(),
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            &mut reference,
            &source,
            alpha,
            beta,
        )
        .unwrap();
    }
    Replays { dense, reference }
}

/// One, a scale that is neither 1 nor -1, both signed zeros, and a genuinely
/// complex one (its real part on `f64`).
fn alphas<T: TestScalar>() -> Vec<T> {
    vec![
        T::from_parts(1.0, 0.0),
        T::from_parts(-2.5, 0.0),
        T::from_parts(0.0, 0.0),
        T::from_parts(-0.0, -0.0),
        T::from_parts(0.7, -0.3),
    ]
}

fn check<T>() -> bool
where
    T: TestScalar
        + TreeTransformScalar
        + RecouplingCoefficientAction<f64>
        + DenseRecouplingScalar
        + ConjugateValue,
{
    let recoupling: Vec<&str> = recoupling_fixtures().iter().map(|f| f.name).collect();
    // The scalar-loop strided driver and the GEMM path round differently; a
    // recoupling fixture where they differ bitwise shows the backend ran the
    // GEMM path.
    let mut paths_differ = false;
    for fixture in all_fixtures() {
        let source = payload::<T>(&fixture);
        let initial = initial::<T>(&fixture);
        for overwrite in [true, false] {
            for alpha in alphas::<T>() {
                let r = replay::<T>(&fixture, overwrite, alpha);
                let expected = fixture.expected_scaled(&source, &initial, overwrite, alpha);
                assert_eq!(r.dense.len(), expected.len(), "{}", fixture.name);
                for (index, (a, b)) in r.dense.iter().zip(&expected).enumerate() {
                    assert!(
                        a.distance(*b) <= 1e-12,
                        "{} / {} / overwrite={overwrite} / alpha={alpha:?}: \
                         element {index} is {a:?}, oracle {b:?}",
                        fixture.name,
                        T::NAME,
                    );
                }
                if recoupling.contains(&fixture.name) {
                    paths_differ |= r.dense != r.reference;
                }
            }
        }
    }
    paths_differ
}

#[test]
fn dense_backend_is_the_batched_gemm_path() {
    // What: the production batched-GEMM replay equals the explicit-index
    // semantics on every shared fixture (degeneracy > 1, several blocks,
    // Single and Multi recoupling groups, conjugated sources, inactive and
    // zero-extent layouts), in both destination modes, for every caller
    // scale and both payload dtypes.
    let real = check::<f64>();
    let complex = check::<Complex64>();
    assert!(
        real || complex,
        "no fixture separates the GEMM path from the scalar oracle bitwise"
    );
}
