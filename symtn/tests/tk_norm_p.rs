//! TensorKit `norm(t, p)` correspondence for `TensorMap::norm` (issue #597,
//! item 2).
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

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Complex64, Error};
use symtn::typed::{GradedSpace, TensorMap};

use crate::numerics;

/// Floating terms per norm: one per stored entry of the fixtures (21). The
/// two engines sum the same terms in different orders, so exact equality is
/// not the claim; the `numerics` bound is far tighter than any weighting or
/// dispatch mistake could survive.
const TERMS: usize = 21;

use crate::fixtures;

use fixtures::host_runtime;

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

fn typed_u1() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let rule = Arc::new(U1FusionRule);
    let pairs = |entries: [(i32, usize); 3]| {
        GradedSpace::try_new(
            Arc::clone(&rule),
            entries.map(|(charge, deg)| (U1Irrep::new(charge), deg)),
        )
        .unwrap()
    };
    (
        pairs([(-1, 2), (0, 3), (1, 4)]),
        pairs([(-1, 5), (0, 1), (1, 2)]),
    )
}

fn typed_su2() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let rule = Arc::new(SU2FusionRule);
    let pairs = |entries: [(usize, usize); 3]| {
        GradedSpace::try_new(
            Arc::clone(&rule),
            entries.map(|(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin), deg)),
        )
        .unwrap()
    };
    (
        pairs([(0, 2), (1, 3), (2, 4)]),
        pairs([(0, 5), (1, 1), (2, 2)]),
    )
}

#[test]
fn typed_norm_p_matches_tensorkit() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (u1_v, u1_w) = typed_u1();
    let (su2_v, su2_w) = typed_su2();

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
    let (v, w) = typed_u1();
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

/// The compact arm reads `Σ_c k_c` stored values with the same `dim(c)`
/// weights, so it must agree with the dense answer on a materialized twin.
/// (The `O(Σ_c k_c)` / zero-materialization half of the claim is asserted by
/// byte counting in `typed_diagonal_allocations.rs`; this is the value half.)
#[test]
fn compact_norm_p_equals_the_dense_answer() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (v, w) = typed_su2();
    let dense: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&v], [&w], |_, indices| real_fill(indices))
            .unwrap();
    let s = dense.svd_compact(engine, &[0], &[1]).unwrap().s;

    // `0 * id + 1 * s` is a value-identical *dense* twin of `s`: `id` on the
    // bond space is dense, so the sum takes the dense route without disturbing
    // `s`'s own storage.
    let twin = TensorMap::isomorphism(engine, &rt, &s.domain(), &s.domain())
        .unwrap()
        .axpby(engine, 0.0, &s, 1.0)
        .unwrap();

    for p in [1.0, 2.0, 3.0, 0.5, f64::INFINITY] {
        numerics::assert_close(
            &format!("compact vs dense p={p}"),
            s.norm(engine, p).unwrap(),
            twin.norm(engine, p).unwrap(),
            TERMS,
        );
    }
}

// --- `norm(t, Inf)` on non-finite payloads (issue #1286) ------------------
//
// TensorKit reduces with `mapreduce(max, …)` (`src/tensors/linalg.jl:_norm`,
// revision cfaa073) and Julia's `max` returns NaN when either argument is NaN.
// Rust's `f64::max` is IEEE `maxNum`, which *discards* NaN — folding with it
// would let a poisoned tensor report a finite magnitude, which is exactly what
// overflow control in a long contraction sequence reads this value to detect.
// QSpace has no max-abs production helper (concrete absence).
//
// The poison sits in the *last* coupled sector below, so a fold that dropped
// NaN would still return the finite maximum of the earlier blocks.

/// The coupled sector poisoned by the fixtures below: the `+1` block is last in
/// the U(1) bond's engine order and is not where the clean maximum lives.
fn poisoned(trees: &symtn::typed::BlockFusionTrees<U1Irrep>) -> bool {
    *trees.coupled() == U1Irrep::new(1)
}

fn u1_real_with(value: f64) -> TensorMap<U1FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (v, w) = typed_u1();
    TensorMap::from_subblock_fn(engine, &rt, [&v], [&w], |trees, indices| {
        if poisoned(trees) && indices == [0, 0] {
            value
        } else {
            real_fill(indices)
        }
    })
    .unwrap()
}

#[test]
fn norm_inf_propagates_nan_hidden_in_the_imaginary_part() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (v, w) = typed_u1();
    let tensor: TensorMap<U1FusionRule, Complex64> =
        TensorMap::from_subblock_fn(engine, &rt, [&v], [&w], |trees, indices| {
            if poisoned(trees) && indices == [0, 0] {
                // Finite real part: only |z| = hypot(re, im) sees the NaN.
                Complex64::new(1.0, f64::NAN)
            } else {
                complex_fill(indices)
            }
        })
        .unwrap();
    assert!(
        tensor.norm(engine, f64::INFINITY).unwrap().is_nan(),
        "c64 NaN in Im(z)"
    );
}

#[test]
fn norm_inf_propagates_nan_through_compact_and_lazy_storage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = host_runtime();
    let (bond, _) = typed_u1();
    let compact = TensorMap::<_, f64>::diagonal(
        engine,
        &rt,
        &bond,
        [
            symtn::typed::SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![2.0, 3.0],
            },
            symtn::typed::SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, 4.0, 5.0],
            },
            symtn::typed::SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![f64::NAN, 1.0, 1.0, 1.0],
            },
        ],
    )
    .unwrap();
    assert!(
        compact.norm(engine, f64::INFINITY).unwrap().is_nan(),
        "the compact arm reads the stored spectrum and must fold it the same way"
    );

    // `adjoint` on dense storage is a lazy parent-backed view, and `norm(Inf)`
    // reads the parent payload without materializing it: a third fold site.
    let lazy = u1_real_with(f64::NAN).adjoint(engine).unwrap();
    assert!(
        lazy.norm(engine, f64::INFINITY).unwrap().is_nan(),
        "lazy adjoint arm"
    );
}
