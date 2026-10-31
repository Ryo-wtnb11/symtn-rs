//! `TensorMap::norm(p)` is the single norm entry.
//!
//! Every arm is checked against TensorKit's formula evaluated by hand from
//! the fill functions: `max |x|` for `p = ∞`, else
//! `(Σ_c dim(c) Σ |x|^p)^(1/p)` (`src/tensors/linalg.jl`), over dense, compact,
//! lazy-adjoint, single-precision and out-of-range (rescaled) payloads.
//! Recorded TensorKit values of the same fixtures are asserted in
//! `tk_norm_p.rs` and `tk_norm_range.rs`.

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Complex32, Complex64, Error, Runtime};
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

use crate::legs::{su2_norm_legs, u1_norm_legs};

fn real_fill(indices: &[usize]) -> f64 {
    1.0 + 0.5 * indices[0] as f64 - 0.25 * indices[1] as f64
}

fn complex_fill(indices: &[usize]) -> Complex64 {
    Complex64::new(
        real_fill(indices),
        0.5 + 0.125 * indices[0] as f64 + 0.375 * indices[1] as f64,
    )
}

fn compact_su2(rt: &Runtime) -> TensorMap<SU2FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (bond, _) = su2_norm_legs();
    let spectrum = |twice_spin, values: &[f64]| SectorSpectrum {
        sector: SU2Irrep::from_twice_spin(twice_spin),
        values: values.to_vec(),
    };
    TensorMap::diagonal(
        engine,
        rt,
        &bond,
        [
            spectrum(0, &[3.25, -0.5]),
            spectrum(1, &[1.75, 0.0, -2.125]),
            spectrum(2, &[0.375, 4.5, -1.0, 0.625]),
        ],
    )
    .unwrap()
}

/// Exponents checked for every tensor.
const EXPONENTS: [f64; 5] = [2.0, f64::INFINITY, 1.0, 3.0, 0.5];

/// `(dim(c), rows, cols)` of every coupled block of a `[V] ← [W]` pair:
/// the sectors both legs carry, with their quantum dimensions.
fn u1_blocks() -> Vec<(f64, usize, usize)> {
    vec![(1.0, 2, 5), (1.0, 3, 1), (1.0, 4, 2)]
}

fn su2_blocks() -> Vec<(f64, usize, usize)> {
    vec![(1.0, 2, 5), (2.0, 3, 1), (3.0, 4, 2)]
}

/// TensorKit `norm(t, p)` by hand: `max |x|` for `p = ∞`, else
/// `(Σ_c dim(c) Σ |x|^p)^(1/p)` over `(dim(c), entries of block c)`.
fn hand_norm(blocks: &[(f64, Vec<f64>)], p: f64) -> f64 {
    let entries = || {
        blocks
            .iter()
            .flat_map(|(dim, values)| values.iter().map(move |value| (*dim, value.abs())))
    };
    if p.is_infinite() {
        return entries().map(|(_, value)| value).fold(0.0, f64::max);
    }
    entries()
        .map(|(dim, value)| dim * value.powf(p))
        .sum::<f64>()
        .powf(p.recip())
}

/// The magnitudes of `fill(i, j)` over each `rows × cols` block.
fn filled(
    blocks: Vec<(f64, usize, usize)>,
    magnitude: impl Fn(&[usize]) -> f64,
) -> Vec<(f64, Vec<f64>)> {
    blocks
        .into_iter()
        .map(|(dim, rows, cols)| {
            let values = (0..rows)
                .flat_map(|i| (0..cols).map(move |j| [i, j]))
                .map(|ij| magnitude(&ij))
                .collect();
            (dim, values)
        })
        .collect()
}

fn assert_hand(
    name: &str,
    blocks: &[(f64, Vec<f64>)],
    scale: f64,
    tol: f64,
    mut norm: impl FnMut(f64) -> Result<f64, Error>,
) {
    for p in EXPONENTS {
        let actual = norm(p).unwrap();
        let expected = scale * hand_norm(blocks, p);
        assert!(
            (actual - expected).abs() <= tol * expected,
            "{name} norm({p}) = {actual:e}, expected {expected:e}"
        );
    }
}

#[test]
fn every_arm_matches_the_dimension_weighted_entrywise_norm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (uv, uw) = u1_norm_legs();
    let (sv, sw) = su2_norm_legs();
    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&uv], [&uw], |_, i| real_fill(i)).unwrap();
    let u1_c64: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &rt, [&uv], [&uw], |_, i| complex_fill(i)).unwrap();
    let su2_f64: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&sv], [&sw], |_, i| real_fill(i)).unwrap();
    let su2_c64: TensorMap<SU2FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &rt, [&sv], [&sw], |_, i| complex_fill(i)).unwrap();
    // Every fill entry is dyadic, so the `f32` payload holds it exactly.
    let su2_c32: TensorMap<SU2FusionRule, Complex32> =
        TensorMap::from_subblock_fn(engine, &rt, [&sv], [&sw], |_, i| {
            let z = complex_fill(i);
            Complex32::new(z.re as f32, z.im as f32)
        })
        .unwrap();
    // Squares leave the f64 range in both directions, so `p = 2` and the
    // power sums take their rescaling passes; the norm scales linearly.
    let huge: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&sv], [&sw], |_, i| 1e200 * real_fill(i))
            .unwrap();
    let tiny: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &rt, [&uv], [&uw], |_, i| 1e-200 * complex_fill(i))
            .unwrap();
    let compact = compact_su2(&rt);
    let adjoint = su2_c64.adjoint(engine).unwrap();

    let real = |i: &[usize]| real_fill(i).abs();
    let modulus = |i: &[usize]| complex_fill(i).norm();
    let (u1_real, u1_complex) = (filled(u1_blocks(), real), filled(u1_blocks(), modulus));
    let (su2_real, su2_complex) = (filled(su2_blocks(), real), filled(su2_blocks(), modulus));
    let diagonal = vec![
        (1.0, vec![3.25, 0.5]),
        (2.0, vec![1.75, 0.0, 2.125]),
        (3.0, vec![0.375, 4.5, 1.0, 0.625]),
    ];
    let tol = 1e-14;
    assert_hand("u1_f64", &u1_real, 1.0, tol, |p| u1_f64.norm(engine, p));
    assert_hand("u1_c64", &u1_complex, 1.0, tol, |p| u1_c64.norm(engine, p));
    assert_hand("su2_f64", &su2_real, 1.0, tol, |p| su2_f64.norm(engine, p));
    assert_hand("su2_c64", &su2_complex, 1.0, tol, |p| {
        su2_c64.norm(engine, p)
    });
    assert_hand("su2_c32", &su2_complex, 1.0, 1e-6, |p| {
        su2_c32.norm(engine, p)
    });
    assert_hand("huge", &su2_real, 1e200, tol, |p| huge.norm(engine, p));
    assert_hand("tiny", &u1_complex, 1e-200, tol, |p| tiny.norm(engine, p));
    assert_hand("compact", &diagonal, 1.0, tol, |p| compact.norm(engine, p));
    assert_hand("adjoint", &su2_complex, 1.0, tol, |p| {
        adjoint.norm(engine, p)
    });
}

#[test]
fn infinity_arm_is_positive_zero_without_entries() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (uv, uw) = u1_norm_legs();
    let only_zero = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let only_one = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 3)]).unwrap();
    let empty: TensorMap<U1FusionRule, f64> =
        TensorMap::zeros(engine, &rt, [&only_zero], [&only_one]).unwrap();
    let zeros: TensorMap<U1FusionRule, f64> = TensorMap::zeros(engine, &rt, [&uv], [&uw]).unwrap();
    for (what, tensor) in [("empty", &empty), ("all-zero", &zeros)] {
        let value = tensor.norm(engine, f64::INFINITY).unwrap();
        assert_eq!(
            value.to_bits(),
            0.0_f64.to_bits(),
            "{what}: {value:e} is not +0.0"
        );
    }
}

#[test]
fn invalid_exponents_are_typed_errors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (uv, uw) = u1_norm_legs();
    let tensor: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&uv], [&uw], |_, i| real_fill(i)).unwrap();
    for p in [f64::NAN, f64::NEG_INFINITY, 0.0, -0.0, -1.0] {
        assert!(
            matches!(tensor.norm(engine, p), Err(Error::InvalidArgument(_))),
            "norm({p})"
        );
    }
}

/// The device reduction exists only for `p = 2`. Its value is the square root
/// of the device self inner product, exactly the body of the removed
/// argument-free device `norm()`.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_norm_is_the_frobenius_arm_and_rejects_other_exponents() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let rt = Runtime::builder().build().unwrap();
    let (uv, uw) = u1_norm_legs();
    let (sv, sw) = su2_norm_legs();
    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&uv], [&uw], |_, i| real_fill(i)).unwrap();
    let su2_c64: TensorMap<SU2FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &rt, [&sv], [&sw], |_, i| complex_fill(i)).unwrap();

    let real = u1_f64.to_cuda(cuda).unwrap();
    let complex = su2_c64.to_cuda(cuda).unwrap();
    let real_norm = real.norm(cuda, 2.0).unwrap();
    let complex_norm = complex.norm(cuda, 2.0).unwrap();
    assert_eq!(
        real_norm.to_bits(),
        real.inner(cuda, &real).unwrap().sqrt().to_bits()
    );
    assert_eq!(
        complex_norm.to_bits(),
        complex.inner(cuda, &complex).unwrap().re.sqrt().to_bits()
    );
    assert_eq!(
        complex
            .adjoint(cuda)
            .unwrap()
            .norm(cuda, 2.0)
            .unwrap()
            .to_bits(),
        complex_norm.to_bits()
    );
    let host = hand_norm(&filled(u1_blocks(), |i| real_fill(i).abs()), 2.0);
    assert!((real_norm - host).abs() <= 1e-13 * host);
    let host = hand_norm(&filled(su2_blocks(), |i| complex_fill(i).norm()), 2.0);
    assert!((complex_norm - host).abs() <= 1e-13 * host);

    for p in [f64::INFINITY, 1.0, 3.0, 0.5] {
        assert!(
            matches!(real.norm(cuda, p), Err(Error::UnsupportedOnDevice(_))),
            "device norm({p})"
        );
    }
    for p in [f64::NAN, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(
            matches!(real.norm(cuda, p), Err(Error::InvalidArgument(_))),
            "device norm({p})"
        );
    }
}

/// A compact Host operand reduces in `O(Σ_c k_c)` and uploads dense; the
/// device Frobenius arm then gives the same value as the Host compact arm and
/// the hand oracle `Σ_c dim(c) Σ_k conj(a_k) b_k`, and every other exponent
/// stays `UnsupportedOnDevice` on the uploaded compact operand.
/// Device storage is multiplicity-free only, so there is no checked
/// Generic device row.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn compact_host_reductions_match_the_device_frobenius_arm() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let rt = Runtime::builder().build().unwrap();
    let (bond, _) = su2_norm_legs();
    let value = |twice: usize, k: usize, shift: f64| {
        Complex64::new(
            1.0 + 0.5 * k as f64 - shift * twice as f64,
            shift + 0.25 * k as f64,
        )
    };
    let spectrum = |shift: f64| {
        [(0, 2), (1, 3), (2, 4)].map(|(twice, degeneracy)| SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(twice),
            values: (0..degeneracy).map(|k| value(twice, k, shift)).collect(),
        })
    };
    let a = TensorMap::<_, Complex64>::diagonal(engine, &rt, &bond, spectrum(0.3)).unwrap();
    let b = TensorMap::<_, Complex64>::diagonal(engine, &rt, &bond, spectrum(-0.2)).unwrap();
    let (mut norm2, mut inner) = (0.0, Complex64::default());
    for (twice, degeneracy) in [(0, 2), (1, 3), (2, 4)] {
        let dim = (twice + 1) as f64;
        for k in 0..degeneracy {
            norm2 += dim * value(twice, k, 0.3).norm_sqr();
            inner += dim * value(twice, k, 0.3).conj() * value(twice, k, -0.2);
        }
    }
    let close = |what: &str, got: Complex64, want: Complex64| {
        assert!(
            (got - want).norm() <= 1e-13 * want.norm(),
            "{what}: {got} against {want}"
        );
    };
    let (device_a, device_b) = (a.to_cuda(cuda).unwrap(), b.to_cuda(cuda).unwrap());
    close(
        "host compact norm",
        a.norm(engine, 2.0).unwrap().into(),
        norm2.sqrt().into(),
    );
    close(
        "device norm",
        device_a.norm(cuda, 2.0).unwrap().into(),
        norm2.sqrt().into(),
    );
    close("host compact inner", a.inner(engine, &b).unwrap(), inner);
    close(
        "device inner",
        device_a.inner(cuda, &device_b).unwrap(),
        inner,
    );
    for p in [1.0, 3.0, f64::INFINITY] {
        assert!(a.norm(engine, p).is_ok(), "host compact norm({p})");
        assert!(
            matches!(device_a.norm(cuda, p), Err(Error::UnsupportedOnDevice(_))),
            "device norm({p})"
        );
    }
}
