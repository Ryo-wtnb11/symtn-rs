//! Lazy-adjoint payload reads go through `FusionOperand` plus the shared
//! strided owner. Expected values are a per-element adjoint kernel written
//! out literally (`common::literal_adjoint_payload`) and TensorKit's `tr`
//! definition (`common::literal_weighted_trace`); neither touches the
//! production adjoint kernel.

use crate::common;

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{BlockFusionTrees, GradedSpace, TensorMap};

use common::{assert_same_tensor, literal_adjoint_payload, literal_weighted_trace};

fn close_c64(a: Complex64, b: Complex64) -> bool {
    (a - b).norm() <= 1e-12 * (1.0 + b.norm())
}

fn close_f64(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

fn fingerprint<S: std::fmt::Debug>(trees: &BlockFusionTrees<S>) -> f64 {
    (format!("{trees:?}").bytes().fold(0u32, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(byte as u32)
    }) % 97) as f64
}

fn complex_value<S: std::fmt::Debug>(trees: &BlockFusionTrees<S>, indices: &[usize]) -> Complex64 {
    let tree = fingerprint(trees);
    let mut re = tree;
    let mut im = -0.5 * tree + 0.25;
    for (axis, &index) in indices.iter().enumerate() {
        re += (index as f64 + 1.0) * (axis as f64 + 1.0);
        im += (index as f64 + 1.0) * (axis as f64 + 1.0) * (axis as f64 + 1.0) * 0.5;
    }
    Complex64::new(re, im)
}

fn u1_leg(
    provider: &Arc<U1FusionRule>,
    sectors: &[(i32, usize)],
    dual: bool,
) -> GradedSpace<U1FusionRule> {
    let space = GradedSpace::try_new(
        Arc::clone(provider),
        sectors
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    if dual {
        space.try_dual().unwrap()
    } else {
        space
    }
}

/// Rank 3+1 U(1) fixture with a dual leg, non-uniform degeneracies and several
/// coupled sectors.
fn u1_rank4(runtime: &Runtime) -> TensorMap<U1FusionRule, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let a = u1_leg(&provider, &[(-1, 2), (0, 1), (1, 3)], false);
    let b = u1_leg(&provider, &[(0, 2), (1, 1)], true);
    let c = u1_leg(&provider, &[(-1, 1), (1, 2)], false);
    let d = u1_leg(
        &provider,
        &[(-2, 2), (-1, 3), (0, 1), (1, 2), (2, 1)],
        false,
    );
    let tensor =
        TensorMap::from_subblock_fn(engine, runtime, [&a, &b, &c], [&d], complex_value).unwrap();
    assert!(tensor.subblock_count() >= 4);
    tensor
}

/// SU(2) 2+2 fixture with several coupled sectors and degeneracies 2/1/3.
fn su2_rank4(runtime: &Runtime) -> TensorMap<SU2FusionRule, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
            (SU2Irrep::from_twice_spin(2), 3),
        ],
    )
    .unwrap();
    let other = GradedSpace::try_new(
        provider,
        [
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(3), 1),
        ],
    )
    .unwrap();
    let tensor =
        TensorMap::from_subblock_fn(engine, runtime, [&leg, &other], [&leg, &leg], complex_value)
            .unwrap();
    let coupled: std::collections::BTreeSet<_> = (0..tensor.subblock_count())
        .map(|index| *tensor.subblock_fusion_trees(index).unwrap().coupled())
        .collect();
    assert!(
        coupled.len() >= 2,
        "fixture must span several coupled sectors"
    );
    tensor
}

macro_rules! assert_lazy_adjoint_reads_and_transforms_match_literal {
    ($engine:ident, $parent:expr, $conj:expr, $close:expr) => {{
        let parent = $parent;
        let lazy = parent.adjoint($engine).unwrap();
        let parent_snapshot = snapshot!($engine, parent);
        let lazy_snapshot = snapshot!($engine, lazy);
        let literal = literal_adjoint_payload(&parent_snapshot, &lazy_snapshot, $conj);
        assert_eq!(
            literal.len(),
            lazy.materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len()
        );
        for (actual, expected) in lazy
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(&literal)
        {
            assert!($close(*actual, *expected), "{actual:?} != {expected:?}");
        }

        // An Owned tensor carrying the literal payload, placed by trees.
        let codomain = lazy.codomain();
        let domain = lazy.domain();
        let owned = TensorMap::from_subblock_fn(
            $engine,
            parent.runtime(),
            &codomain,
            &domain,
            |trees, indices| {
                let (_, geometry) = lazy_snapshot
                    .blocks
                    .iter()
                    .find(|(candidate, _)| candidate == trees)
                    .unwrap();
                literal[common::linear(geometry, indices)]
            },
        )
        .unwrap();
        assert_same_tensor(&snapshot!($engine, owned), &lazy_snapshot, $close);

        let rank = lazy.rank();
        let split = lazy.codomain_rank();
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.rotate_left(1);
        axes.swap(0, 1);
        let (codomain_axes, domain_axes) = axes.split_at(split);
        let new_split = if split > 1 { split - 1 } else { split + 1 };
        let levels: Vec<usize> = (0..rank).map(|axis| (axis * 7) % rank).collect();
        // Planar transpose: rotate the cyclic order (codomain, reversed domain).
        let mut cycle: Vec<usize> = (0..split).chain((split..rank).rev()).collect();
        cycle.rotate_left(1);
        let (cyclic_codomain, reversed_domain) = cycle.split_at(split);
        let cyclic_domain: Vec<usize> = reversed_domain.iter().rev().copied().collect();
        // TensorKit's argument-free `transpose`: the full planar rotation.
        let full_codomain: Vec<usize> = (split..rank).rev().collect();
        let full_domain: Vec<usize> = (0..split).rev().collect();
        let pairs: [(TensorMap<_, _>, TensorMap<_, _>); 5] = [
            (
                lazy.permute($engine, codomain_axes, domain_axes).unwrap(),
                owned.permute($engine, codomain_axes, domain_axes).unwrap(),
            ),
            (
                lazy.braid($engine, codomain_axes, domain_axes, &levels)
                    .unwrap(),
                owned
                    .braid($engine, codomain_axes, domain_axes, &levels)
                    .unwrap(),
            ),
            (
                lazy.repartition($engine, new_split).unwrap(),
                owned.repartition($engine, new_split).unwrap(),
            ),
            (
                lazy.transpose($engine, &full_codomain, &full_domain)
                    .unwrap(),
                owned
                    .transpose($engine, &full_codomain, &full_domain)
                    .unwrap(),
            ),
            (
                lazy.transpose($engine, cyclic_codomain, &cyclic_domain)
                    .unwrap(),
                owned
                    .transpose($engine, cyclic_codomain, &cyclic_domain)
                    .unwrap(),
            ),
        ];
        for (actual, expected) in &pairs {
            assert_same_tensor(
                &snapshot!($engine, actual),
                &snapshot!($engine, expected),
                $close,
            );
        }
    }};
}

#[test]
fn u1_rank4_lazy_adjoint_matches_the_literal_kernel_for_real_and_complex() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let complex = u1_rank4(&runtime);
    assert!(complex
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im != 0.0));
    assert_lazy_adjoint_reads_and_transforms_match_literal!(
        engine,
        complex.clone(),
        |z: Complex64| z.conj(),
        close_c64
    );
    assert_lazy_adjoint_reads_and_transforms_match_literal!(
        engine,
        complex.re(engine).unwrap(),
        |x: f64| x,
        close_f64
    );
}

#[test]
fn su2_rank4_lazy_adjoint_matches_the_literal_kernel_for_real_and_complex() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let complex = su2_rank4(&runtime);
    assert!(complex
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im != 0.0));
    assert_lazy_adjoint_reads_and_transforms_match_literal!(
        engine,
        complex.clone(),
        |z: Complex64| z.conj(),
        close_c64
    );
    assert_lazy_adjoint_reads_and_transforms_match_literal!(
        engine,
        complex.re(engine).unwrap(),
        |x: f64| x,
        close_f64
    );
}

#[test]
fn empty_support_lazy_adjoint_materializes_an_empty_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let codomain = u1_leg(&provider, &[(1, 2)], false);
    let domain = u1_leg(&provider, &[(0, 3)], false);
    let parent: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&codomain], [&domain], |_, _| 1.0).unwrap();
    assert_eq!(parent.subblock_count(), 0);
    let lazy = parent.adjoint(engine).unwrap();
    assert_eq!(lazy.subblock_count(), 0);
    assert!(lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(lazy
        .transpose(engine, &[1], &[0])
        .unwrap()
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
}

fn su2_dim(sector: &SU2Irrep) -> f64 {
    (sector.twice_spin() + 1) as f64
}

#[test]
fn su2_tr_matches_the_literal_weighted_diagonal_sum_and_conjugates_lazily() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
            (SU2Irrep::from_twice_spin(2), 3),
        ],
    )
    .unwrap();
    let other = GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 2)]).unwrap();
    let complex = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &other],
        [&leg, &other],
        complex_value,
    )
    .unwrap();
    let real = complex.re(engine).unwrap();

    let mut expected = Complex64::new(0.0, 0.0);
    literal_weighted_trace(&snapshot!(engine, complex), su2_dim, |value, weight| {
        expected += value * weight
    });
    assert!(
        expected.im.abs() > 1e-6,
        "fixture trace must be genuinely complex"
    );
    assert!(close_c64(complex.tr(engine).unwrap(), expected));
    let lazy = complex.adjoint(engine).unwrap();
    assert!(close_c64(
        lazy.tr(engine).unwrap(),
        complex.tr(engine).unwrap().conj()
    ));

    let mut expected_real = 0.0;
    literal_weighted_trace(&snapshot!(engine, real), su2_dim, |value, weight| {
        expected_real += value * weight
    });
    assert!(close_f64(real.tr(engine).unwrap(), expected_real));
    assert!(close_f64(
        real.adjoint(engine).unwrap().tr(engine).unwrap(),
        expected_real
    ));

    // Unweighted diagonal sum differs: the dim(c) weight is observable.
    let mut unweighted = 0.0;
    literal_weighted_trace(
        &snapshot!(engine, real),
        |_| 1.0,
        |value, weight| unweighted += value * weight,
    );
    assert!((unweighted - expected_real).abs() > 1e-6);

    let non_endomorphism = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &other],
        [&leg, &leg],
        complex_value,
    )
    .unwrap();
    assert!(matches!(
        non_endomorphism.tr(engine).unwrap_err(),
        symtn::typed::Error::Operation(operation)
            if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert!(matches!(
        non_endomorphism.adjoint(engine).unwrap().tr(engine).unwrap_err(),
        symtn::typed::Error::Operation(operation)
            if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
}

#[test]
fn lazy_adjoint_materialization_copies_non_finite_values_bit_exactly() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // The owner's `alpha == 1, beta == 0` path is a plain (conjugating) copy:
    // `1 * (inf + 0i)` would turn the imaginary part into NaN, which an
    // element-wise odometer never does. `conj(inf + 0i) == inf - 0i`, sign kept.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let a = u1_leg(&provider, &[(-1, 2), (0, 1), (1, 2)], false);
    let d = u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)], true);
    let complex = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&a],
        [&a, &d],
        |trees, indices| match indices.iter().sum::<usize>() % 3 {
            0 => Complex64::new(f64::INFINITY, 0.0),
            1 => Complex64::new(-1.5, f64::NEG_INFINITY),
            _ => complex_value(trees, indices),
        },
    )
    .unwrap();
    let lazy = complex.adjoint(engine).unwrap();
    let literal = literal_adjoint_payload(
        &snapshot!(engine, complex),
        &snapshot!(engine, lazy),
        |z: Complex64| z.conj(),
    );
    assert!(literal
        .iter()
        .any(|z| z.re == f64::INFINITY && z.im == 0.0 && z.im.is_sign_negative()));
    assert!(literal.iter().any(|z| z.im == f64::INFINITY));
    for (actual, expected) in lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(&literal)
    {
        assert_eq!(
            actual.re.to_bits(),
            expected.re.to_bits(),
            "{actual:?} != {expected:?}"
        );
        assert_eq!(
            actual.im.to_bits(),
            expected.im.to_bits(),
            "{actual:?} != {expected:?}"
        );
    }

    let real = TensorMap::from_subblock_fn(engine, &runtime, [&a], [&a, &d], |_, indices| {
        match indices.iter().sum::<usize>() % 3 {
            0 => f64::INFINITY,
            1 => f64::NEG_INFINITY,
            _ => indices.iter().sum::<usize>() as f64 + 0.5,
        }
    })
    .unwrap();
    let lazy = real.adjoint(engine).unwrap();
    let literal = literal_adjoint_payload(
        &snapshot!(engine, real),
        &snapshot!(engine, lazy),
        |x: f64| x,
    );
    assert!(literal.contains(&f64::INFINITY) && literal.contains(&f64::NEG_INFINITY));
    for (actual, expected) in lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(&literal)
    {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
