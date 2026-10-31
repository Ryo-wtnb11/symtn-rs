//! Host transforms keep a real structural coefficient real (#1398).
//!
//! Oracle: a real coefficient acting on a complex block is, by definition, the
//! same coefficient acting separately on the real and the imaginary component.
//! So every case transforms one `Complex64` tensor and the two `f64` tensors
//! that hold its components, and compares the results bit for bit. The oracle
//! never reads a symtn coefficient, a descriptor or a kernel, so it is
//! independent of the code under test; it is TensorKit's `ComplexF64 *
//! Float64` in `stridedtensoradd!`, which Julia evaluates componentwise.
//!
//! The fixtures carry `-0`, `±inf`, NaN payloads and subnormals, which is what
//! makes the comparison sharp: the complex product `(c + 0i)(x + yi)` differs
//! from the componentwise `(cx, cy)` exactly on those values, because `0 * y`
//! is a NaN for a non-finite `y` and `cx - 0` loses the sign of a zero.

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use symtn::typed::Engine;

use num_complex::{Complex32, Complex64};
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::BlockFusionTrees;
use symtn::typed::{GradedSpace, Runtime, TensorMap};

/// A deterministic component per `(tree pair, index, salt)`, with signed
/// zeros, infinities, NaN payloads and subnormals mixed into ordinary values.
fn component<S: Hash>(trees: &BlockFusionTrees<S>, indices: &[usize], salt: u8) -> f64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    (
        trees.coupled(),
        trees.codomain_uncoupled(),
        trees.codomain_innerlines(),
        trees.codomain_vertices(),
        trees.domain_uncoupled(),
        trees.domain_innerlines(),
        trees.domain_vertices(),
        indices,
        salt,
    )
        .hash(&mut hasher);
    let bits = hasher.finish();
    const SPECIAL: [u64; 6] = [
        0x8000_0000_0000_0000, // -0
        0x7ff0_0000_0000_0000, // +inf
        0xfff0_0000_0000_0000, // -inf
        0x7ff8_0000_0000_1398, // NaN with a payload
        0xfff8_0000_0000_0042, // negative NaN with a payload
        0x0000_0000_0000_0001, // smallest subnormal
    ];
    // Keyed on the degeneracy index rather than the hash, so every fixture
    // covers all six payloads instead of covering them on average.
    let selector = (indices.iter().copied().sum::<usize>() + usize::from(salt)) % 12;
    if selector < SPECIAL.len() {
        return f64::from_bits(SPECIAL[selector]);
    }
    let magnitude = 0.5 + (bits >> 11) as f64 / (1u64 << 53) as f64;
    let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
    sign * magnitude * f64::powi(2.0, ((bits >> 1) % 16) as i32 - 8)
}

/// Transforms the complex tensor and its two real components the same way,
/// and asserts the complex result is the componentwise real result.
macro_rules! assert_componentwise {
    ($engine:ident, [$($codomain:expr),* $(,)?], [$($domain:expr),* $(,)?], |$t:ident| $body:expr) => {
        assert_componentwise!($engine,
            Complex64, f64, [$($codomain),*], [$($domain),*], |$t| $body
        )
    };
    ($engine:ident, $complex:ty, $real:ty, [$($codomain:expr),* $(,)?], [$($domain:expr),* $(,)?],
     |$t:ident| $body:expr) => {{
        let runtime = Runtime::builder().build().unwrap();
        let codomain = vec![$($codomain),*];
        let domain = vec![$($domain),*];
        let build = |$engine: &mut dyn Engine, f: &dyn Fn(&BlockFusionTrees<_>, &[usize]) -> $real| {
            TensorMap::<_, $real>::from_subblock_fn($engine,
                &runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                |trees, indices| f(trees, indices),
            )
            .unwrap()
        };
        let complex = TensorMap::<_, $complex>::from_subblock_fn($engine,
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |trees, indices| {
                <$complex>::new(
                    component(trees, indices, 0) as $real,
                    component(trees, indices, 1) as $real,
                )
            },
        )
        .unwrap();
        let real = build($engine, &|trees, indices| component(trees, indices, 0) as $real);
        let imaginary = build($engine, &|trees, indices| component(trees, indices, 1) as $real);
        assert_non_trivial(
            complex.subblock_count(),
            complex
                .dense_data().unwrap()
                .iter()
                .flat_map(|value| [f64::from(value.re), f64::from(value.im)]),
        );

        let got = { let $t = &complex; $body };
        let want_real = { let $t = &real; $body };
        let want_imaginary = { let $t = &imaginary; $body };
        let (got, want_real, want_imaginary) = (
            got.unwrap(),
            want_real.unwrap(),
            want_imaginary.unwrap(),
        );
        assert_eq!(got.dense_data().unwrap().len(), want_real.dense_data().unwrap().len());
        assert_eq!(got.dense_data().unwrap().len(), want_imaginary.dense_data().unwrap().len());
        for (position, value) in got.dense_data().unwrap().iter().enumerate() {
            assert_eq!(
                (value.re.to_bits(), value.im.to_bits()),
                (
                    want_real.dense_data().unwrap()[position].to_bits(),
                    want_imaginary.dense_data().unwrap()[position].to_bits(),
                ),
                "element {position} of {} <- {}",
                stringify!([$($codomain),*]),
                stringify!([$($domain),*]),
            );
        }
        (complex, got)
    }};
}

/// The fixture must actually exercise the values that separate a componentwise
/// scale from a complex multiply, and must have more than one block.
fn assert_non_trivial(block_count: usize, mut components: impl Iterator<Item = f64> + Clone) {
    assert!(block_count > 1, "fixture needs several blocks");
    assert!(
        components.clone().any(|value| !value.is_finite()),
        "no ±inf or NaN"
    );
    assert!(
        components.any(|value| value == 0.0 && value.is_sign_negative()),
        "no -0"
    );
    // Subnormals are in the value set but are not asserted per fixture: unlike
    // a signed zero or a non-finite value they survive a complex multiply, so
    // they do not discriminate, and the smallest blocks here do not reach them.
}

/// The negative control for the oracle above: a full complex multiply by an
/// exact real 1 does not reproduce these payloads, so a transform that
/// restores `alpha * value` must fail every case in this file.
#[test]
fn a_complex_multiply_by_one_cannot_reproduce_the_payloads() {
    for value in [
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(-0.0, -0.0),
        Complex64::new(f64::from_bits(0x7ff8_0000_0000_1398), 1.0),
    ] {
        let product = Complex64::new(1.0, 0.0) * value;
        assert_ne!(
            (product.re.to_bits(), product.im.to_bits()),
            (value.re.to_bits(), value.im.to_bits()),
            "{value} survived a complex multiply by one"
        );
    }
}

fn u1(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2(provider: &Arc<SU2FusionRule>, pairs: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

fn bits(data: &[Complex64]) -> Vec<(u64, u64)> {
    data.iter()
        .map(|v| (v.re.to_bits(), v.im.to_bits()))
        .collect()
}

#[test]
fn u1_transforms_scale_componentwise() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 3), (0, 2), (1, 4)]);
    let dual = leg.try_dual().unwrap();
    let other = u1(&provider, &[(-1, 1), (0, 3), (1, 2)]);

    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.permute(
        engine,
        &[1, 0],
        &[2]
    ));
    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t
        .repartition(engine, 1));
    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.transpose(
        engine,
        &[2],
        &[1, 0]
    ));
    assert_componentwise!(engine, [&leg], [&other, &dual], |t| t.permute(
        engine,
        &[2, 0],
        &[1]
    ));
    // c32: the coefficient converts with `as f32` before acting, still
    // componentwise.
    assert_componentwise!(engine, Complex32, f32, [&leg, &other], [&dual], |t| t
        .permute(engine, &[1, 0], &[2]));
    assert_componentwise!(engine, Complex32, f32, [&leg, &other], [&dual], |t| t
        .repartition(engine, 1));

    // A round trip composes to the identity coefficient, so it must return the
    // input bits unchanged.
    let (source, _) = assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.permute(
        engine,
        &[1, 0],
        &[2]
    ));
    let back = source
        .permute(engine, &[1, 0], &[2])
        .unwrap()
        .permute(engine, &[1, 0], &[2])
        .unwrap();
    assert_eq!(
        bits(back.dense_data().unwrap()),
        bits(source.dense_data().unwrap())
    );
}

#[test]
fn su2_transforms_scale_componentwise() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 2), (1, 3), (2, 1)]);
    let dual = leg.try_dual().unwrap();
    let other = su2(&provider, &[(0, 1), (1, 2)]);

    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.permute(
        engine,
        &[1, 0],
        &[2]
    ));
    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t
        .repartition(engine, 1));
    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.transpose(
        engine,
        &[2],
        &[1, 0]
    ));
    assert_componentwise!(engine, [&leg], [&other], |t| t.repartition(engine, 0));
}

#[test]
fn u1_times_su2_transforms_scale_componentwise() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap(),
    );
    let label = |charge: i32, twice_spin: usize| {
        product_sector(U1Irrep::new(charge), SU2Irrep::from_twice_spin(twice_spin))
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(label(0, 0), 2), (label(1, 1), 3), (label(-1, 1), 2)],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let other = GradedSpace::try_new(
        Arc::clone(&provider),
        [(label(0, 0), 1), (label(1, 1), 2), (label(0, 2), 2)],
    )
    .unwrap();

    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.permute(
        engine,
        &[1, 0],
        &[2]
    ));
    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t
        .repartition(engine, 1));
    assert_componentwise!(engine, [&leg, &dual], [&other], |t| t.transpose(
        engine,
        &[2],
        &[1, 0]
    ));
}

/// fZ2 x U(1): the permutation of two odd legs contributes a fermionic `-1`,
/// which must stay a real sign rather than a complex multiply.
#[test]
fn fermionic_transforms_scale_componentwise() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 3),
            (label(true, 1), 2),
            (label(true, -1), 2),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let other = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 2),
            (label(true, 1), 1),
            (label(false, 1), 3),
        ],
    )
    .unwrap();

    let (source, permuted) = assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.permute(
        engine,
        &[1, 0],
        &[2]
    ));
    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t
        .repartition(engine, 1));
    assert_componentwise!(engine, [&leg, &other], [&dual], |t| t.transpose(
        engine,
        &[2],
        &[1, 0]
    ));
    assert_componentwise!(engine, [&leg, &dual], [&other, &leg], |t| t.permute(
        engine,
        &[1, 2],
        &[3, 0]
    ));

    // The swap sign is real and involutive: permuting back is the exact input.
    let back = permuted.permute(engine, &[1, 0], &[2]).unwrap();
    assert_eq!(
        bits(back.dense_data().unwrap()),
        bits(source.dense_data().unwrap())
    );
}

/// Trace (#1407): with `α = 1` the structural coefficient acts on the traced
/// sum in its own type (TensorKit folds `α′ = α * coeff` but applies it per
/// element before the reduction; symtn applies it after the sum). The SU(2) partial trace carries
/// `dim(coupled) / dim(uncoupled)` and the fermionic one a twist `-1`, so
/// both reach a non-unit real coefficient.
#[test]
fn traces_scale_componentwise() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 3), (0, 2), (1, 4)]);
    let dual = leg.try_dual().unwrap();
    let other = u1(&provider, &[(-1, 1), (0, 3), (1, 2)]);
    assert_componentwise!(engine, [&leg, &other], [&leg, &other], |t| t
        .trace_pairs(engine, &[(0, 2)]));
    assert_componentwise!(engine, [&dual, &other], [&dual, &other], |t| t
        .trace_pairs(engine, &[(1, 3)]));
    assert_componentwise!(
        engine,
        Complex32,
        f32,
        [&leg, &other],
        [&leg, &other],
        |t| t.trace_pairs(engine, &[(1, 3)])
    );

    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 2), (1, 3), (2, 1)]);
    let dual = leg.try_dual().unwrap();
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    assert_componentwise!(engine, [&leg, &other], [&leg, &other], |t| t
        .trace_pairs(engine, &[(1, 3)]));
    assert_componentwise!(engine, [&other, &dual], [&other, &dual], |t| t
        .trace_pairs(engine, &[(0, 2)]));
    assert_componentwise!(
        engine,
        Complex32,
        f32,
        [&leg, &other],
        [&leg, &other],
        |t| t.trace_pairs(engine, &[(1, 3)])
    );

    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 3),
            (label(true, 1), 2),
            (label(true, -1), 2),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    assert_componentwise!(engine, [&leg, &dual], [&leg, &dual], |t| t
        .trace_pairs(engine, &[(1, 3)]));
    assert_componentwise!(engine, [&leg, &dual], [&leg, &dual], |t| t
        .trace_pairs(engine, &[(0, 2)]));
}

/// Known residual of #1398: a transform that needs a recoupling *matrix* still
/// applies it as a dense GEMM over coefficients promoted to the payload type,
/// so `0 * inf` reappears. Fixing it needs the complex destination block to be
/// driven through the real GEMM as an interleaved real matrix (the column
/// mixing only touches the source index, so the reinterpretation is exact),
/// which belongs to the dense-executor boundary.
#[test]
#[ignore = "multi-block recoupling GEMM still promotes the coefficient (#1398 residual)"]
fn su2_recoupling_matrix_transforms_scale_componentwise() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 2), (1, 3), (2, 1)]);
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    assert_componentwise!(engine, [&leg, &other, &other], [&leg], |t| t.permute(
        engine,
        &[2, 1, 0],
        &[3]
    ));
}
