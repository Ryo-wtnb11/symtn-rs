//! TensorKit `norm(t, p)` correspondence for `TensorMap::norm`.
//!
//! TensorKit 0.17 `src/tensors/linalg.jl:257-275`:
//!
//! ```text
//! p == Inf     -> maximum entry magnitude over blocks(t)
//! finite p > 0 -> (sum_c dim(c) * norm(block_c, p)^p)^(1/p)
//! otherwise    -> ArgumentError
//! ```
//!
//! `norm(block, p)` is Julia's *entrywise* p-norm (matrices included), so only
//! the multiset of a coupled sector's entries and its quantum dimension enter
//! the result — never the block's shape or the engine's fusion-tree ordering.
//! That is what makes the oracle below portable between TensorKit's storage and
//! symtn's.
//!
//! # Oracle provenance
//!
//! The constants were checked against TensorKit 0.17.1 on Julia 1.11.6, from
//! a tensor built block-by-block with
//! exactly the fill this file uses:
//!
//! ```julia
//! realfill(i, j) = 1.0 + 0.5 * (i - 1) - 0.25 * (j - 1)
//! cplxfill(i, j) = realfill(i, j) + im * (0.5 + 0.125 * (i - 1) + 0.375 * (j - 1))
//! t = zeros(T, V <- W); for (c, b) in blocks(t), j in axes(b, 2), i in axes(b, 1)
//!     b[i, j] = fill(i, j)
//! end
//! norm(t, p)
//! ```
//!
//! with `V = U1Space(-1 => 2, 0 => 3, 1 => 4)`, `W = U1Space(-1 => 5, 0 => 1, 1 => 2)`
//! and the SU(2) pair `V = SU2Space(0 => 2, 1//2 => 3, 1 => 4)`,
//! `W = SU2Space(0 => 5, 1//2 => 1, 1 => 2)`.
//!
//! The degeneracies are deliberately all distinct, so no two coupled sectors
//! carry the same entry multiset: a `dim(c)` weight attached to the wrong
//! sector changes the answer.

use symtn::sector::{SU2FusionRule, U1FusionRule};
use symtn::typed::TensorMap;
use symtn::typed::{Complex64, Error};

use crate::numerics;

/// Floating terms per norm: one per stored entry of the fixtures (21). The
/// two engines sum the same terms in different orders, so exact equality is
/// not the claim; the `numerics` bound is far tighter than any weighting or
/// dispatch mistake could survive.
const TERMS: usize = 21;

use crate::fixtures;

use fixtures::host_runtime;

use crate::legs::{su2_norm_legs, u1_norm_legs};

/// The real fill, in the engine's 0-based degeneracy coordinates.
///
/// `1.0 + 0.5 * i - 0.25 * j` hits exactly zero at `(0, 4)`, which the
/// 5-fold domain sector reaches — so the fixtures also cover a `0^p` term.
fn real_fill(indices: &[usize]) -> f64 {
    1.0 + 0.5 * indices[0] as f64 - 0.25 * indices[1] as f64
}

fn complex_fill(indices: &[usize]) -> Complex64 {
    Complex64::new(
        real_fill(indices),
        0.5 + 0.125 * indices[0] as f64 + 0.375 * indices[1] as f64,
    )
}

// TensorKit 0.17.0 oracle values, indexed as [p = 1, 2, 3, Inf].
const U1_F64: [f64; 4] = [25.0, 6.2048368229954285, 4.080_404_211_522_764, 2.5];
const U1_C64: [f64; 4] = [
    36.088_144_488_042_53,
    8.145_167_278_822_456,
    5.0601391475683535,
    2.6487025125521364,
];
const SU2_F64: [f64; 4] = [55.5, 9.656_603_957_913_982, 5.588_779_616_741_119, 2.5];
const SU2_C64: [f64; 4] = [
    70.837_746_836_191_09,
    11.637090486887175,
    6.514_050_413_991_05,
    2.6487025125521364,
];

/// The exponents the oracle rows above are indexed by.
const POWERS: [f64; 4] = [1.0, 2.0, 3.0, f64::INFINITY];

#[test]
fn typed_norm_p_matches_tensorkit() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (u1_v, u1_w) = u1_norm_legs();
    let (su2_v, su2_w) = su2_norm_legs();

    let u1_f64: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&u1_v], [&u1_w], |_, indices| {
            real_fill(indices)
        })
        .unwrap();
    let u1_c64: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &rt, [&u1_v], [&u1_w], |_, indices| {
            complex_fill(indices)
        })
        .unwrap();
    let su2_f64: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&su2_v], [&su2_w], |_, indices| {
            real_fill(indices)
        })
        .unwrap();
    let su2_c64: TensorMap<SU2FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &rt, [&su2_v], [&su2_w], |_, indices| {
            complex_fill(indices)
        })
        .unwrap();

    for (&p, &expected) in POWERS.iter().zip(&U1_F64) {
        numerics::assert_close(
            &format!("u1 f64 p={p}"),
            u1_f64.norm(engine, p).unwrap(),
            expected,
            TERMS,
        );
    }
    for (&p, &expected) in POWERS.iter().zip(&U1_C64) {
        numerics::assert_close(
            &format!("u1 c64 p={p}"),
            u1_c64.norm(engine, p).unwrap(),
            expected,
            TERMS,
        );
    }
    for (&p, &expected) in POWERS.iter().zip(&SU2_F64) {
        numerics::assert_close(
            &format!("su2 f64 p={p}"),
            su2_f64.norm(engine, p).unwrap(),
            expected,
            TERMS,
        );
    }
    for (&p, &expected) in POWERS.iter().zip(&SU2_C64) {
        numerics::assert_close(
            &format!("su2 c64 p={p}"),
            su2_c64.norm(engine, p).unwrap(),
            expected,
            TERMS,
        );
    }
}

#[test]
fn typed_norm_p_rejects_non_positive_and_non_finite_p() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (v, w) = u1_norm_legs();
    let tensor: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&v], [&w], |_, indices| real_fill(indices))
            .unwrap();
    for p in [f64::NAN, f64::NEG_INFINITY, 0.0, -1.0, -0.5] {
        assert!(
            matches!(tensor.norm(engine, p), Err(Error::InvalidArgument(_))),
            "norm({p}) must be a typed error, not a panic and not a value"
        );
    }
}
