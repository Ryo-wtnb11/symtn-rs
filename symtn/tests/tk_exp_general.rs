//! Verbatim `%.17g` TensorKit output below. Trimming the literals to the
//! shortest round-tripping form would obscure that provenance for no gain.
#![allow(clippy::excessive_precision)]

//! TensorKit `exp(::AbstractTensorMap)` correspondence for the general
//! (non-Hermitian) arm of `TensorMap::exp` (issue #577).
//!
//! TensorKit 0.17 `src/tensors/linalg.jl`, `exp` at line 44 and `exp!` at
//! lines 420-428:
//!
//! ```text
//! exp(t)  = exp!(copy(t))
//! exp!(t) = (domain(t) == codomain(t) || error(...);
//!            for (c, b) in blocks(t); copy!(b, LinearAlgebra.exp!(b)); end)
//! ```
//!
//! so the only structure in the operation is *blockwise*: the endomorphism
//! check, and then Julia's scaling-and-squaring Padé on each coupled-sector
//! matrix independently. No hermiticity gate, no cross-block coupling.
//!
//! # Oracle provenance
//!
//! The constants below were checked against TensorKit 0.17.1 on Julia 1.11.6,
//! using the same fill as this file:
//!
//! ```julia
//! V = U1Space(0 => 3, 1 => 2)
//! t = zeros(T, V ← V)
//! for (c, b) in blocks(t), j in axes(b, 2), i in axes(b, 1)
//!     re = 0.5 + 0.25 * (i - 1) - 0.75 * (j - 1) + 0.125 * convert(Int, c.charge)
//!     b[i, j] = (T <: Complex) ? complex(re, 0.125 * (i - 1) + 0.375 * (j - 1) - 0.25) : re
//! end
//! b .*= scale
//! norm(t), norm(exp(t))
//! ```
//!
//! # Fixture certification
//!
//! `norm` is a pre-existing method that predates #577, and `norm(t)` is
//! basis-free — `sqrt(Σ_c dim(c) ||block_c||_F²)`, with `dim(c) = 1` for every
//! U(1) sector — so it depends on the multiset of block entries and nothing
//! else. Pinning `norm` of the *input* against Julia therefore proves symtn's
//! fixture is the tensor Julia exponentiated, before any of #577's code runs.
//! The entrywise oracle for the same fixture lives one layer down, in
//! `symtn-matrixalgebra`'s `exp_of_a_multisector_u1_endomorphism_matches_the_series`.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{BlockFusionTrees, GradedSpace, TensorMap};
use symtn::typed::{Complex64, Runtime};

/// Floating terms per norm: one per stored entry of the fixtures (13). The
/// two engines evaluate the same approximant at different Padé degrees for
/// the small blocks (Julia drops to degree 3/5/7/9 below `||A||_1 = 2.1`;
/// symtn always uses [13/13]), so this is approximant agreement, not bitwise
/// agreement, and still far tighter than any coefficient, scaling or dispatch
/// mistake.
const TERMS: usize = 13;

use crate::numerics;

use crate::fixtures;

use fixtures::host_runtime;

fn real_fill(charge: i32, indices: &[usize], scale: f64) -> f64 {
    scale * (0.5 + 0.25 * indices[0] as f64 - 0.75 * indices[1] as f64 + 0.125 * charge as f64)
}

fn imaginary_fill(indices: &[usize], scale: f64) -> f64 {
    scale * (0.125 * indices[0] as f64 + 0.375 * indices[1] as f64 - 0.25)
}

fn typed_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap()
}

fn typed_real(runtime: &Runtime, scale: f64) -> TensorMap<U1FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let leg = typed_space();
    TensorMap::from_subblock_fn(
        engine,
        runtime,
        [&leg],
        [&leg],
        |sectors: &BlockFusionTrees<U1Irrep>, indices: &[usize]| {
            real_fill(sectors.coupled().charge(), indices, scale)
        },
    )
    .unwrap()
}

fn typed_complex(runtime: &Runtime, scale: f64) -> TensorMap<U1FusionRule, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let leg = typed_space();
    TensorMap::from_subblock_fn(
        engine,
        runtime,
        [&leg],
        [&leg],
        |sectors: &BlockFusionTrees<U1Irrep>, indices: &[usize]| {
            Complex64::new(
                real_fill(sectors.coupled().charge(), indices, scale),
                imaginary_fill(indices, scale),
            )
        },
    )
    .unwrap()
}

#[test]
fn general_exp_matches_the_tensorkit_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = host_runtime();

    // (scale, norm(t), norm(exp(t))) from the Julia session quoted above.
    for (scale, input_norm, exponential_norm) in [
        (1.0, 2.2220486043288972, 3.1532168621506798),
        (4.0, 8.8881944173155887, 15.692503963067267),
    ] {
        let typed = typed_real(&runtime, scale);
        numerics::assert_close(
            &format!("f64 scale {scale} typed input fixture"),
            typed.norm(engine, 2.0).unwrap(),
            input_norm,
            TERMS,
        );

        let typed_exp = typed.exp(engine, &[0], &[1]).unwrap();
        numerics::assert_close(
            &format!("f64 scale {scale} exp"),
            typed_exp.norm(engine, 2.0).unwrap(),
            exponential_norm,
            TERMS,
        );
    }

    // c64, where the blocks are non-Hermitian in both parts.
    let typed = typed_complex(&runtime, 1.0);
    numerics::assert_close(
        "c64 typed input fixture",
        typed.norm(engine, 2.0).unwrap(),
        2.5678298230217673,
        TERMS,
    );

    let typed_exp = typed.exp(engine, &[0], &[1]).unwrap();
    numerics::assert_close(
        "c64 exp",
        typed_exp.norm(engine, 2.0).unwrap(),
        3.1806015158373815,
        TERMS,
    );
}

/// `A = [0 1e16; 1e-16 0]`, whose exponential is closed form:
/// `exp(A) = cosh(1) I + sinh(1) A`, because `A² = I`.
const BALANCE_FIXTURE: [f64; 4] = [0.0, 1e-16, 1e16, 0.0];

#[test]
fn general_exp_balances_a_badly_scaled_block_like_julia() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: Julia's `exp!` runs `LAPACK.gebal!('B', A)` *before* the Padé
    // evaluation and undoes it afterwards (stdlib v1.11 `dense.jl:677-782`), so
    // TensorKit parity requires balancing here too. Unbalanced, this block has
    // `||A||_1 = 1e16` and pays 51 squarings, each one amplifying the
    // approximant's error; balanced, its norm is ~1.11 and the approximant is
    // evaluated directly. The exact answer is the same either way, so only the
    // balancing shows up in the values.
    let runtime = host_runtime();
    let space = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let tensor = TensorMap::from_subblock_fn(engine, &runtime, [&space], [&space], |_, indices| {
        BALANCE_FIXTURE[indices[0] + 2 * indices[1]]
    })
    .unwrap();

    let exponential = tensor.exp(engine, &[0], &[1]).unwrap();

    let expected = [
        1.0_f64.cosh(),
        1.0_f64.sinh() * 1e-16,
        1.0_f64.sinh() * 1e16,
        1.0_f64.cosh(),
    ];
    for (index, (&actual, &want)) in exponential
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        let relative = (actual - want).abs() / want.abs();
        assert!(
            relative <= 1e-13,
            "entry {index}: {actual:.17e} differs from {want:.17e} by {relative:e}"
        );
    }
}
