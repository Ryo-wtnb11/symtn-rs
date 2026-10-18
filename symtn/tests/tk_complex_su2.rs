//! Complex SU(2) structural-operation correspondence with TensorKit
//! 0.17.1 (Julia 1.11.6).

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn close(got: f64, expected: f64) {
    assert!((got - expected).abs() <= 1e-10 * expected.abs().max(1.0));
}

fn fill(c0: i64, labels: [i64; 5], idx: &[usize]) -> f64 {
    let [l1, l2, m1, m2, lc] = labels;
    let value = c0
        + 7 * l1
        + 11 * l2
        + 13 * m1
        + 17 * m2
        + 19 * lc
        + 23 * (idx[0] as i64 + 1)
        + 29 * (idx[1] as i64 + 1)
        + 31 * (idx[2] as i64 + 1)
        + 37 * (idx[3] as i64 + 1);
    (value.rem_euclid(41) - 20) as f64
}

#[test]
fn complex_su2_structural_invariants_match_tensorkit() {
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(SU2FusionRule);
    let space = GradedSpace::try_new(
        rule,
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let source: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        [&space, &space],
        [&space, &space],
        |trees, idx| {
            let label = |sector: &SU2Irrep| sector.twice_spin() as i64;
            let codomain = trees.codomain_uncoupled();
            let domain = trees.domain_uncoupled();
            let labels = [
                label(&codomain[0]),
                label(&codomain[1]),
                label(&domain[0]),
                label(&domain[1]),
                label(trees.coupled()),
            ];
            Complex64::new(fill(11, labels, idx), fill(17, labels, idx) / 3.0)
        },
    )
    .unwrap();
    let permuted = source.permute(&[1, 0], &[3, 2]).unwrap();
    let gram = source.adjoint().unwrap().compose(&source).unwrap();

    for (tensor, expected_norm, expected_trace) in [
        (&source, 40.741_733_994_626_31, Complex64::new(-81.0, -9.0)),
        (
            &permuted,
            40.741_733_994_626_32,
            Complex64::new(-81.000_000_000_000_04, -9.000_000_000_000_004),
        ),
        (
            &gram,
            853.858_236_997_74,
            Complex64::new(1_659.888_888_888_889, 0.0),
        ),
    ] {
        close(tensor.norm(2.0).unwrap(), expected_norm);
        let trace = tensor.tr().unwrap();
        close(trace.re, expected_trace.re);
        close(trace.im, expected_trace.im);
    }

    let mut singular_values: Vec<_> = permuted
        .svd_vals(&[0, 1], &[2, 3])
        .unwrap()
        .iter()
        .flat_map(|spectrum| spectrum.values.iter().copied())
        .collect();
    singular_values.sort_by(|left, right| right.partial_cmp(left).unwrap());
    let expected = [
        23.892354784323476,
        21.02264361151887,
        8.027729719194868,
        3.221_429_043_489_522,
        0.845_649_010_956_779_1,
    ];
    assert_eq!(singular_values.len(), expected.len());
    for (got, expected) in singular_values.into_iter().zip(expected) {
        close(got, expected);
    }
}
