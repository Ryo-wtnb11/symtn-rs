use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::BlockFusionTrees;
use symtn::typed::OperationError;
use symtn::typed::{Complex64, GradedSpace, PhysicalDense, PhysicalDenseError, Runtime, TensorMap};

fn su2_multitree<D: symtn::typed::TensorScalar>(
    runtime: &Runtime,
    value: impl Fn(&[SU2Irrep]) -> D,
) -> TensorMap<SU2FusionRule, D> {
    let half = SU2Irrep::from_twice_spin(1);
    let leg = GradedSpace::try_new(std::sync::Arc::new(SU2FusionRule), [(half, 1)]).unwrap();
    TensorMap::from_subblock_fn(runtime, [&leg, &leg, &leg], [&leg], |trees, _| {
        value(trees.codomain_innerlines())
    })
    .unwrap()
}

fn assert_complex_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).norm() < 2.0e-12);
    }
}

fn assert_real_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2.0e-12);
    }
}

#[test]
fn physical_dense_roundtrips_real_and_complex_su2_multitree_data() {
    let runtime = Runtime::builder().build().unwrap();
    let real = su2_multitree(&runtime, |inner| {
        if inner[0].twice_spin() == 0 {
            1.25
        } else {
            -0.75
        }
    });
    let real_reduced = real.dense_data().unwrap().to_vec();
    let physical = real.to_physical_dense().unwrap();
    assert_eq!(physical.shape, [2, 2, 2, 2]);
    assert_eq!(real.dense_data().unwrap(), real_reduced);
    assert_real_close(
        real.project_physical_dense(&physical)
            .unwrap()
            .dense_data()
            .unwrap(),
        &real_reduced,
    );

    let complex = su2_multitree(&runtime, |inner| {
        if inner[0].twice_spin() == 0 {
            Complex64::new(1.25, -0.5)
        } else {
            Complex64::new(-0.75, 0.25)
        }
    });
    let complex_reduced = complex.dense_data().unwrap().to_vec();
    let physical = complex.to_physical_dense().unwrap();
    assert_eq!(physical.shape, [2, 2, 2, 2]);
    let projected = complex.project_physical_dense(&physical).unwrap();
    assert_complex_close(projected.dense_data().unwrap(), &complex_reduced);
    assert_eq!(complex.dense_data().unwrap(), complex_reduced);
}

#[test]
fn projection_validates_shape_and_length_without_changing_target() {
    let runtime = Runtime::builder().build().unwrap();
    let target = su2_multitree(&runtime, |_| 9.0);
    let before = target.dense_data().unwrap().to_vec();

    let bad_shape = PhysicalDense {
        shape: vec![2, 2],
        data: vec![0.0; 4],
    };
    let error = target.project_physical_dense(&bad_shape).unwrap_err();
    assert!(matches!(
        error,
        PhysicalDenseError::Operation(OperationError::ShapeMismatch { .. })
    ));
    assert_eq!(target.dense_data().unwrap(), before);

    let bad_length = PhysicalDense {
        shape: vec![2, 2, 2, 2],
        data: vec![0.0; 15],
    };
    let error = target.project_physical_dense(&bad_length).unwrap_err();
    assert!(matches!(
        error,
        PhysicalDenseError::Operation(OperationError::ElementCountMismatch { .. })
    ));
    assert_eq!(target.dense_data().unwrap(), before);
}

#[test]
fn physical_entries_match_tensorkit_su2_values() {
    let runtime = Runtime::builder().build().unwrap();
    let real = su2_multitree(&runtime, |inner| f64::from(inner[0].twice_spin() == 2));
    let inv_sqrt_6 = 1.0 / 6.0_f64.sqrt();
    let expected_real = [
        0.0,
        -inv_sqrt_6,
        -inv_sqrt_6,
        0.0,
        2.0 * inv_sqrt_6,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -2.0 * inv_sqrt_6,
        0.0,
        inv_sqrt_6,
        inv_sqrt_6,
        0.0,
    ];
    assert_real_close(&real.to_physical_dense().unwrap().data, &expected_real);

    let scale = Complex64::new(1.0, 2.0);
    let complex = su2_multitree(&runtime, |inner| {
        if inner[0].twice_spin() == 0 {
            scale
        } else {
            Complex64::new(0.0, 0.0)
        }
    });
    let inv_sqrt_2 = 1.0 / 2.0_f64.sqrt();
    let mut expected_complex = vec![Complex64::new(0.0, 0.0); 16];
    for (index, coefficient) in [
        (1, -inv_sqrt_2),
        (2, inv_sqrt_2),
        (13, -inv_sqrt_2),
        (14, inv_sqrt_2),
    ] {
        expected_complex[index] = scale * coefficient;
    }
    assert_complex_close(
        &complex.to_physical_dense().unwrap().data,
        &expected_complex,
    );
}

fn permute_each_axis<D: Copy>(physical: &PhysicalDense<D>, target_to_source: &[usize]) -> Vec<D> {
    assert_eq!(physical.shape, [target_to_source.len(); 2]);
    let dimension = target_to_source.len();
    (0..dimension * dimension)
        .map(|target| {
            let i = target % dimension;
            let j = target / dimension;
            physical.data[target_to_source[i] + dimension * target_to_source[j]]
        })
        .collect()
}

#[test]
fn su2_spin_one_singlet_projects_to_explicit_doubled_u1_charges() {
    let runtime = Runtime::builder().build().unwrap();
    let spin_one = SU2Irrep::from_twice_spin(2);
    let su2_leg =
        GradedSpace::try_new(std::sync::Arc::new(SU2FusionRule), [(spin_one, 1)]).unwrap();
    let singlet: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg, &su2_leg], [], |_, _| 1.0).unwrap();
    let su2_physical = singlet.to_physical_dense().unwrap();
    assert_eq!(su2_physical.shape, [3, 3]);

    // SU(2) uses doubled magnetic labels (2, 0, -2), whereas this U(1)
    // leg's TensorKit order is (0, 2, -2). Keep the basis change explicit.
    let target_to_source = [1, 0, 2];
    let u1_physical = PhysicalDense {
        shape: vec![3, 3],
        data: permute_each_axis(&su2_physical, &target_to_source),
    };
    let u1_leg = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 1),
            (U1Irrep::new(-2), 1),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let target: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&u1_leg, &u1_leg], []).unwrap();
    let projected = target.project_physical_dense(&u1_physical).unwrap();

    let inv_sqrt_3 = 1.0 / 3.0_f64.sqrt();
    let mut labelled = projected
        .subblocks()
        .unwrap()
        .map(|(trees, block)| {
            (
                (
                    trees.codomain_uncoupled()[0].charge(),
                    trees.codomain_uncoupled()[1].charge(),
                ),
                *block.get(&[0, 0]).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    labelled.sort_unstable_by_key(|entry| entry.0);
    let expected = [
        ((-2, 2), inv_sqrt_3),
        ((0, 0), -inv_sqrt_3),
        ((2, -2), inv_sqrt_3),
    ];
    assert_eq!(labelled.len(), expected.len());
    for ((labels, actual), (expected_labels, expected_value)) in labelled.into_iter().zip(expected)
    {
        assert_eq!(labels, expected_labels);
        assert!((actual - expected_value).abs() < 2.0e-12);
    }
    assert_real_close(
        &projected.to_physical_dense().unwrap().data,
        &u1_physical.data,
    );
}

/// Arbitrary label-dependent fill of one reduced element, so that every
/// sector, inner line, coupled sector and degeneracy index is distinguishable
/// in the dense array. Returns `(re, im)`.
fn fill(labels: &[f64], inner: f64, coupled: f64, index: &[usize]) -> (f64, f64) {
    let mut value = 1.0 + coupled / 5.0 + inner / 11.0;
    for (k, (&label, &index)) in labels.iter().zip(index).enumerate() {
        let k = (k + 1) as f64;
        value += k * (label + 3.0) / 7.0 + index as f64 * (k + 1.0) / 3.0;
    }
    (value, value / 2.0 - labels.iter().sum::<f64>() / 13.0)
}

/// Sets each reduced element of `trees` to [`fill`].
fn fixture_value<S, D: From<f64>>(
    trees: &BlockFusionTrees<S>,
    index: &[usize],
    label: impl Fn(&S) -> f64,
    complex: impl Fn(f64, f64) -> D,
) -> D {
    let labels = trees
        .codomain_uncoupled()
        .iter()
        .chain(trees.domain_uncoupled())
        .map(&label)
        .collect::<Vec<_>>();
    let inner = trees
        .codomain_innerlines()
        .iter()
        .chain(trees.domain_innerlines())
        .map(&label)
        .sum::<f64>();
    let (re, im) = fill(&labels, inner, label(trees.coupled()), index);
    complex(re, im)
}

/// Physical basis of one leg as `(label, degeneracy index, m index)` in
/// TensorKit order: sectors in the given order, and within a sector the
/// magnetic index runs fastest (`kron(fusiontensor, block)`).
fn leg_basis(sectors: &[(i32, usize, usize)]) -> Vec<(i32, usize, usize)> {
    sectors
        .iter()
        .flat_map(|&(label, degeneracy, dim)| {
            (0..degeneracy).flat_map(move |n| (0..dim).map(move |m| (label, n, m)))
        })
        .collect()
}

/// Expands a zero-based column-major linear index over `legs`.
fn leg_entries(legs: &[Vec<(i32, usize, usize)>], mut linear: usize) -> Vec<(i32, usize, usize)> {
    legs.iter()
        .map(|leg| {
            let entry = leg[linear % leg.len()];
            linear /= leg.len();
            entry
        })
        .collect()
}

fn factorial(n: i32) -> f64 {
    (1..=n).map(f64::from).product()
}

/// Condon-Shortley Clebsch-Gordan coefficient `<j1 m1; j2 m2 | j m>` from
/// Racah's closed form, with doubled spins and projections. This is the
/// TensorKitSectors `SU2Irrep` `fusiontensor` gauge (WignerSymbols.jl
/// `clebschgordan`), computed independently of the symtn/fusion-symbols code
/// under test.
fn clebsch_gordan(j1: i32, m1: i32, j2: i32, m2: i32, j: i32, m: i32) -> f64 {
    if m1 + m2 != m
        || m.abs() > j
        || m1.abs() > j1
        || m2.abs() > j2
        || j < (j1 - j2).abs()
        || j > j1 + j2
        || (j1 + j2 + j) % 2 != 0
    {
        return 0.0;
    }
    let half = |doubled: i32| factorial(doubled / 2);
    let prefactor = (f64::from(j + 1) * half(j1 + j2 - j) * half(j1 - j2 + j) * half(j2 - j1 + j)
        / half(j1 + j2 + j + 2)
        * half(j + m)
        * half(j - m)
        * half(j1 - m1)
        * half(j1 + m1)
        * half(j2 - m2)
        * half(j2 + m2))
    .sqrt();
    let sum = (0..=(j1 + j2 - j) / 2)
        .filter_map(|k| {
            let args = [
                j1 + j2 - j - 2 * k,
                j1 - m1 - 2 * k,
                j2 + m2 - 2 * k,
                j - j2 + m1 + 2 * k,
                j - j1 - m2 + 2 * k,
            ];
            args.iter().all(|&a| a >= 0).then(|| {
                f64::from(1 - 2 * (k % 2)) / (factorial(k) * args.map(half).iter().product::<f64>())
            })
        })
        .sum::<f64>();
    prefactor * sum
}

#[test]
fn clebsch_gordan_oracle_matches_textbook_values() {
    // <1/2 1/2; 1/2 -1/2 | 0 0> = 1/sqrt2, <1 1; 1 -1 | 0 0> = 1/sqrt3,
    // <1 0; 1/2 1/2 | 1/2 1/2> = -1/sqrt3.
    assert!((clebsch_gordan(1, 1, 1, -1, 0, 0) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-15);
    assert!((clebsch_gordan(2, 2, 2, -2, 0, 0) - 1.0 / 3.0_f64.sqrt()).abs() < 1e-15);
    assert!((clebsch_gordan(2, 0, 1, 1, 1, 1) + 1.0 / 3.0_f64.sqrt()).abs() < 1e-15);
}

/// Column-major `permutedims(data, axes)`.
fn permute_dense<D: Copy>(physical: &PhysicalDense<D>, axes: &[usize]) -> PhysicalDense<D> {
    let shape = axes.iter().map(|&a| physical.shape[a]).collect::<Vec<_>>();
    let mut source_strides = vec![1; physical.shape.len()];
    for a in 1..physical.shape.len() {
        source_strides[a] = source_strides[a - 1] * physical.shape[a - 1];
    }
    let data = (0..physical.data.len())
        .map(|mut target| {
            let mut source = 0;
            for (&a, &dimension) in axes.iter().zip(&shape) {
                source += (target % dimension) * source_strides[a];
                target /= dimension;
            }
            physical.data[source]
        })
        .collect();
    PhysicalDense { shape, data }
}

const PERMUTATIONS: [(&[usize], &[usize]); 6] = [
    (&[0, 1, 2, 3], &[]),
    (&[0], &[1, 2, 3]),
    (&[], &[0, 1, 2, 3]),
    (&[3, 0, 1], &[2]),
    (&[1, 2], &[3, 0]),
    (&[3, 2], &[1, 0]),
];

/// Expands `$tensor`, checks that its permutations are plain axis
/// permutations of the dense array and that projection inverts the
/// expansion, and returns `(shape, data)` as complex values.
macro_rules! checked_physical_dense {
    ($tensor:expr, $to_complex:expr) => {{
        let (tensor, to_complex) = (&$tensor, $to_complex);
        let physical = tensor.to_physical_dense().unwrap();

        // TensorKit's order makes bending or braiding a leg a plain axis
        // permutation of the dense array for these bosonic providers
        // (TensorKit 0.17.1 `convert(Array, permute(t, p))` has the same property).
        for (codomain, domain) in PERMUTATIONS {
            let axes = [codomain, domain].concat();
            let permuted = tensor.permute(codomain, domain).unwrap();
            let actual = permuted.to_physical_dense().unwrap();
            let expected = permute_dense(&physical, &axes);
            assert_eq!(actual.shape, expected.shape, "{axes:?}");
            let actual = actual
                .data
                .iter()
                .map(|&x| to_complex(x))
                .collect::<Vec<_>>();
            let expected = expected
                .data
                .iter()
                .map(|&x| to_complex(x))
                .collect::<Vec<_>>();
            assert_complex_close(&actual, &expected);
        }

        let reduced = tensor.dense_data().unwrap().to_vec();
        let projected = tensor.project_physical_dense(&physical).unwrap();
        let projected = projected
            .dense_data()
            .unwrap()
            .iter()
            .map(|&x| to_complex(x));
        let reduced = reduced.into_iter().map(&to_complex).collect::<Vec<_>>();
        assert_complex_close(&projected.collect::<Vec<_>>(), &reduced);

        let data = physical
            .data
            .iter()
            .map(|&x| to_complex(x))
            .collect::<Vec<_>>();
        (physical.shape, data)
    }};
}

/// U(1) is abelian with trivial fusion tensors and cups, so the dense array of
/// `V ⊗ V' ← V' ⊗ V` is the reduced element at the charges of each axis,
/// nonzero iff codomain and domain charges agree. TensorKit orders U(1)
/// sectors as `0, 1, -1` (by `|q|`, positive first), and the dual `V'` carries
/// `dual(c) = -c` at `axes(V, c)`.
fn u1_expected(complex: bool) -> Vec<Complex64> {
    let v = leg_basis(&[(0, 2, 1), (1, 1, 1), (-1, 1, 1)]);
    let dual = leg_basis(&[(0, 2, 1), (-1, 1, 1), (1, 1, 1)]);
    let legs = [v.clone(), dual.clone(), dual, v];
    (0..256)
        .map(|linear| {
            let entries = leg_entries(&legs, linear);
            let labels = entries.iter().map(|e| f64::from(e.0)).collect::<Vec<_>>();
            let index = entries.iter().map(|e| e.1).collect::<Vec<_>>();
            let coupled = labels[0] + labels[1];
            if coupled != labels[2] + labels[3] {
                return Complex64::new(0.0, 0.0);
            }
            let (re, im) = fill(&labels, 0.0, coupled, &index);
            Complex64::new(re, if complex { im } else { 0.0 })
        })
        .collect()
}

#[test]
fn u1_dense_order_matches_tensorkit_for_dual_and_bent_legs() {
    let runtime = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let dual = v.try_dual().unwrap();
    let label = |q: &U1Irrep| f64::from(q.charge());
    let real = TensorMap::<U1FusionRule, f64>::from_subblock_fn(
        &runtime,
        [&v, &dual],
        [&dual, &v],
        |trees, index| fixture_value(trees, index, label, |re, _| re),
    )
    .unwrap();
    let (shape, actual) = checked_physical_dense!(real, |x| Complex64::new(x, 0.0));
    assert_eq!(shape, [4, 4, 4, 4]);
    assert_complex_close(&actual, &u1_expected(false));
    let complex = TensorMap::<U1FusionRule, Complex64>::from_subblock_fn(
        &runtime,
        [&v, &dual],
        [&dual, &v],
        |trees, index| fixture_value(trees, index, label, Complex64::new),
    )
    .unwrap();
    let (shape, actual) = checked_physical_dense!(complex, |x| x);
    assert_eq!(shape, [4, 4, 4, 4]);
    assert_complex_close(&actual, &u1_expected(true));
}

fn su2_space(degeneracies: [usize; 3]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        std::sync::Arc::new(SU2FusionRule),
        [0, 1, 2].map(|spin| (SU2Irrep::from_twice_spin(spin), degeneracies[spin])),
    )
    .unwrap()
}

/// Dense CG expansion of `W ⊗ W ⊗ W ← W` with `W = 0 ⊕ (1/2)^2 ⊕ 1` (no dual
/// legs; the degenerate spin-1/2 sector pins the `m`-fastest axis order). The
/// codomain tree `((a1 a2)_e a3)_c` is `C^{a1 a2}_{e} C^{e a3}_{c}`, the
/// one-leg domain tree is `δ(m4, m_c)`, and each tree contributes its reduced
/// element times that fusion tensor (TensorKit `convert(Array, t)`).
fn su2_non_dual_expected(complex: bool) -> Vec<Complex64> {
    let w = leg_basis(&[(0, 1, 1), (1, 2, 2), (2, 1, 3)]);
    let legs = [w.clone(), w.clone(), w.clone(), w];
    (0..8usize.pow(4))
        .map(|linear| {
            let entries = leg_entries(&legs, linear);
            let spins = entries.iter().map(|e| e.0).collect::<Vec<_>>();
            let m = entries
                .iter()
                .map(|&(spin, _, m)| spin - 2 * m as i32)
                .collect::<Vec<_>>();
            let labels = spins.iter().map(|&s| f64::from(s)).collect::<Vec<_>>();
            let index = entries.iter().map(|e| e.1).collect::<Vec<_>>();
            let (a1, a2, a3, c) = (spins[0], spins[1], spins[2], spins[3]);
            let mut value = Complex64::new(0.0, 0.0);
            for e in ((a1 - a2).abs()..=a1 + a2).step_by(2) {
                let coefficient = clebsch_gordan(a1, m[0], a2, m[1], e, m[0] + m[1])
                    * clebsch_gordan(e, m[0] + m[1], a3, m[2], c, m[3]);
                if coefficient != 0.0 {
                    let (re, im) = fill(&labels, f64::from(e), f64::from(c), &index);
                    value += Complex64::new(re, if complex { im } else { 0.0 }) * coefficient;
                }
            }
            value
        })
        .collect()
}

#[test]
fn su2_dense_matches_clebsch_gordan_expansion_without_dual_legs() {
    let runtime = Runtime::builder().build().unwrap();
    let w = su2_space([1, 2, 1]);
    let label = |j: &SU2Irrep| j.twice_spin() as f64;
    let real = TensorMap::<SU2FusionRule, f64>::from_subblock_fn(
        &runtime,
        [&w, &w, &w],
        [&w],
        |trees, index| fixture_value(trees, index, label, |re, _| re),
    )
    .unwrap();
    let (shape, actual) = checked_physical_dense!(real, |x| Complex64::new(x, 0.0));
    assert_eq!(shape, [8, 8, 8, 8]);
    assert_complex_close(&actual, &su2_non_dual_expected(false));
    let complex = TensorMap::<SU2FusionRule, Complex64>::from_subblock_fn(
        &runtime,
        [&w, &w, &w],
        [&w],
        |trees, index| fixture_value(trees, index, label, Complex64::new),
    )
    .unwrap();
    let (shape, actual) = checked_physical_dense!(complex, |x| x);
    assert_eq!(shape, [8, 8, 8, 8]);
    assert_complex_close(&actual, &su2_non_dual_expected(true));
}

/// Sampled entries of `W ⊗ W' ⊗ W ← W'`: the cup/cap signs on the dual legs
/// are not reproduced by the CG oracle above, so these values and the
/// nonzero count were checked against TensorKit 0.17.1 `convert(Array, t)`
/// (column-major, zero-based linear index).
const SU2_DUAL_NONZEROS: usize = 365;
const SU2_DUAL_SAMPLES: [(usize, [f64; 2]); 6] = [
    (16, [4.040_610_178_208_844, 1.911_519_430_460_338]),
    (112, [-4.306_923_121_772_61, -2.044_675_902_242_221]),
    (173, [2.603_245_592_438_400_4, 1.176_007_937_614_934_6]),
    (261, [-2.757_001_442_093_62, -1.252_885_862_442_544_7]),
    (1311, [-1.151_547_754_752_341_8, -0.512_048_713_934_156_3]),
    (1960, [-7.685_714_285_714_284, -3.535_164_835_164_834]),
];

fn assert_su2_dual_samples(actual: &[Complex64], complex: bool) {
    let nonzeros = actual.iter().filter(|x| x.norm() > 1.0e-12).count();
    assert_eq!(nonzeros, SU2_DUAL_NONZEROS);
    for (index, [re, im]) in SU2_DUAL_SAMPLES {
        let expected = Complex64::new(re, if complex { im } else { 0.0 });
        assert!((actual[index] - expected).norm() < 2.0e-12, "{index}");
    }
}

#[test]
fn su2_dense_order_matches_tensorkit_for_dual_and_bent_legs() {
    let runtime = Runtime::builder().build().unwrap();
    let w = su2_space([2, 1, 1]);
    let dual = w.try_dual().unwrap();
    let label = |j: &SU2Irrep| j.twice_spin() as f64;
    let real = TensorMap::<SU2FusionRule, f64>::from_subblock_fn(
        &runtime,
        [&w, &dual, &w],
        [&dual],
        |trees, index| fixture_value(trees, index, label, |re, _| re),
    )
    .unwrap();
    let (shape, actual) = checked_physical_dense!(real, |x| Complex64::new(x, 0.0));
    assert_eq!(shape, [7, 7, 7, 7]);
    assert_su2_dual_samples(&actual, false);
    let complex = TensorMap::<SU2FusionRule, Complex64>::from_subblock_fn(
        &runtime,
        [&w, &dual, &w],
        [&dual],
        |trees, index| fixture_value(trees, index, label, Complex64::new),
    )
    .unwrap();
    let (shape, actual) = checked_physical_dense!(complex, |x| x);
    assert_eq!(shape, [7, 7, 7, 7]);
    assert_su2_dual_samples(&actual, true);
}
