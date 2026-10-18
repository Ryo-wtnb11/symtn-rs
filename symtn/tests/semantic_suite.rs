//! Operation-identity property tests over the user layer (issue #9, part 2)
//! plus TensorKit's weighted rank truncation (part 3).
//!
//! Seeding reuses the splitmix64 scheme of `TensorMap::rand_with_seed`;
//! no new dependencies.
//!
//! Out of scope here (matching the issue): multiplicity `N > 1`,
//! non-symmetric (anyonic) braiding, and repartition/bending — extend this
//! suite when those land.

use std::sync::Arc;
use symtn::typed::ContractSpec;
use symtn::typed::Direction;

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
};
use symtn::sector::{
    Fz2SectorLayout, PackedProductCodec, ProductSectorLayout, Su2SectorLayout, U1SectorLayout,
};
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, Qr, Svd, TensorMap, Truncation};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::{codomain_axes, domain_axes};

type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Su2Codec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type Fz2U1Su2Rule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, Fz2U1Su2Codec>;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

use symtn::typed::Complex32;

/// Floating terms reaching one compared entry of the operation identities: a
/// composition over a contracted `V ⊗ V` leg plus the recoupling terms of a
/// tree transform stay below this count on every fixture.
const TERMS: usize = 64;

/// The cross-library streams compare against TensorKit after long operation
/// chains on magnitudes up to `1e10`; their oracle agreement is a relative
/// envelope `rel * max(1, |want|)`, written as the `numerics` bound with one
/// term and this conditioning.
fn envelope(rel: f64) -> f64 {
    rel / (numerics::K * f64::EPSILON)
}

/// splitmix64, same generator as `TensorMap::rand_with_seed`.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_below(state: &mut u64, bound: usize) -> usize {
    (splitmix64(state) % bound as u64) as usize
}

/// Fisher-Yates permutation of `0..n`.
fn rand_perm(state: &mut u64, n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        perm.swap(i, rand_below(state, i + 1));
    }
    perm
}

macro_rules! for_each_space {
    ($case:ident) => {
        $case!(
            "Z2",
            Arc::new(Z2FusionRule),
            [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)],
            false
        );
        $case!(
            "U1",
            Arc::new(U1FusionRule),
            [
                (U1Irrep::new(-1), 2),
                (U1Irrep::new(0), 2),
                (U1Irrep::new(1), 1),
            ],
            false
        );
        $case!(
            "SU2",
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 2),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
            false
        );
        $case!(
            "fZ2",
            Arc::new(FermionParityFusionRule),
            [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)],
            true
        );
        $case!(
            "fZ2xU1xSU2",
            Arc::new(Fz2U1Su2Rule::new(
                Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
                SU2FusionRule,
            )),
            [
                (
                    product_sector(
                        product_sector(Z2Irrep::EVEN, U1Irrep::new(0)),
                        SU2Irrep::from_twice_spin(0),
                    ),
                    1,
                ),
                (
                    product_sector(
                        product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
                        SU2Irrep::from_twice_spin(1),
                    ),
                    1,
                ),
                (
                    product_sector(
                        product_sector(Z2Irrep::EVEN, U1Irrep::new(2)),
                        SU2Irrep::from_twice_spin(0),
                    ),
                    1,
                ),
            ],
            true
        );
    };
}

// ---------------------------------------------------------------------------
// Permute / braid
// ---------------------------------------------------------------------------

/// permute ∘ permute == permute(composition), random permutation pairs on
/// rank-4 and rank-5 tensors (splits chosen randomly, mirroring the mixed
/// codomain/domain splits exercised by the typed repartition suite).
#[test]
fn permute_composition_law() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0001u64;
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            for (ncod, ndom, seed) in [(2usize, 2usize, 11u64), (1, 4, 12)] {
                let rank = ncod + ndom;
                let cod: Vec<_> = std::iter::repeat(&v).take(ncod).collect();
                let dom: Vec<_> = std::iter::repeat(&v).take(ndom).collect();
                let t: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, cod, dom, seed).unwrap();
                for _ in 0..3 {
                    let s1 = rand_perm(&mut state, rank);
                    let n1 = 1 + rand_below(&mut state, rank - 1);
                    let s2 = rand_perm(&mut state, rank);
                    let n2 = 1 + rand_below(&mut state, rank - 1);
                    let step2 = t
                        .permute(&s1[..n1], &s1[n1..])
                        .unwrap()
                        .permute(&s2[..n2], &s2[n2..])
                        .unwrap();
                    let composed: Vec<usize> = s2.iter().map(|&i| s1[i]).collect();
                    let direct = t.permute(&composed[..n2], &composed[n2..]).unwrap();
                    numerics::assert_slices_close(
                        "composed permutation",
                        step2.dense_data().unwrap(),
                        direct.dense_data().unwrap(),
                        TERMS,
                    );
                    numerics::assert_close(
                        "permutation norm",
                        step2.norm(2.0).unwrap(),
                        t.norm(2.0).unwrap(),
                        TERMS,
                    );
                }
            }
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

/// braid ∘ braid⁻¹ == id: the inverse braid applies the inverse permutation
/// with the levels carried along the strands (TensorKit level semantics:
/// undoing a crossing swaps which strand passes above).
#[test]
fn braid_inverse_roundtrip() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0002u64;
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            let t: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 21).unwrap();
            for _ in 0..3 {
                let s = rand_perm(&mut state, 4);
                let levels = rand_perm(&mut state, 4)
                    .into_iter()
                    .map(|l| l + 1)
                    .collect::<Vec<_>>();
                let braided = t.braid(&s[..2], &s[2..], &levels).unwrap();
                let mut s_inv = vec![0usize; 4];
                for (i, &j) in s.iter().enumerate() {
                    s_inv[j] = i;
                }
                let levels_braided: Vec<usize> = s.iter().map(|&j| levels[j]).collect();
                let back = braided
                    .braid(&s_inv[..2], &s_inv[2..], &levels_braided)
                    .unwrap();
                numerics::assert_slices_close(
                    "braid round trip",
                    back.dense_data().unwrap(),
                    t.dense_data().unwrap(),
                    TERMS,
                );
                numerics::assert_close(
                    "braid norm",
                    braided.norm(2.0).unwrap(),
                    t.norm(2.0).unwrap(),
                    TERMS,
                );
            }
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

/// Bosonic rules: braid == permute for every level assignment.
#[test]
fn bosonic_braid_equals_permute() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0003u64;
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            if !$fermionic {
                let v = GradedSpace::try_new($provider, $pairs).unwrap();
                let t: TensorMap<_, f64> =
                    TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 31).unwrap();
                for _ in 0..3 {
                    let s = rand_perm(&mut state, 4);
                    let levels = rand_perm(&mut state, 4)
                        .into_iter()
                        .map(|l| l + 1)
                        .collect::<Vec<_>>();
                    let braided = t.braid(&s[..2], &s[2..], &levels).unwrap();
                    let permuted = t.permute(&s[..2], &s[2..]).unwrap();
                    numerics::assert_slices_close(
                        "braid equals permute",
                        braided.dense_data().unwrap(),
                        permuted.dense_data().unwrap(),
                        TERMS,
                    );
                }
            }
            let _ = $name;
        }};
    }
    for_each_space!(case);
}

/// Yang-Baxter on three adjacent codomain legs:
/// `b0 b1 b0 == b1 b0 b1` where `b_i` swaps codomain legs `i, i+1`.
/// All rules in scope have real symmetric R (±1), so the crossing
/// chirality (level order) does not affect the value; distinct levels are
/// still passed so the braid engine takes the genuine braiding path.
#[test]
fn yang_baxter_adjacent_swaps() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            let t: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v, &v], [&v], 41).unwrap();
            let swap = |t: &TensorMap<_, f64>, i: usize| {
                let mut cod = vec![0usize, 1, 2];
                cod.swap(i, i + 1);
                t.braid(&cod, &[3], &[1, 2, 3, 4]).unwrap()
            };
            let lhs = swap(&swap(&swap(&t, 0), 1), 0);
            let rhs = swap(&swap(&swap(&t, 1), 0), 1);
            numerics::assert_slices_close(
                "permute identity",
                lhs.dense_data().unwrap(),
                rhs.dense_data().unwrap(),
                TERMS,
            );
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

// ---------------------------------------------------------------------------
// Adjoint / trace / twist / isometry
// ---------------------------------------------------------------------------

/// adjoint is an involution and an antihomomorphism: `(a∘b)† == b†∘a†`.
#[test]
fn adjoint_involution_and_antihomomorphism() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            let a: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 51).unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 52).unwrap();
            numerics::assert_slices_close(
                "double adjoint",
                a.adjoint()
                    .unwrap()
                    .adjoint()
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                a.dense_data().unwrap(),
                TERMS,
            );
            let lhs = a.compose(&b).unwrap().adjoint().unwrap();
            let rhs = b.adjoint().unwrap().compose(&a.adjoint().unwrap()).unwrap();
            numerics::assert_slices_close(
                "adjoint of a composition",
                lhs.materialize().unwrap().dense_data().unwrap(),
                rhs.dense_data().unwrap(),
                TERMS,
            );
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

/// The ordinary matrix trace is cyclic for every supported rule.
#[test]
fn trace_cyclicity() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            for (ncod, seed) in [(1usize, 61u64), (2, 62)] {
                let cod: Vec<_> = std::iter::repeat(&v).take(ncod).collect();
                let a: TensorMap<_, f64> =
                    TensorMap::rand_with_seed(&rt, cod.clone(), cod.clone(), seed).unwrap();
                let b: TensorMap<_, f64> =
                    TensorMap::rand_with_seed(&rt, cod.clone(), cod, seed + 100).unwrap();
                numerics::assert_close(
                    "trace cyclicity",
                    a.compose(&b).unwrap().tr().unwrap(),
                    b.compose(&a).unwrap().tr().unwrap(),
                    TERMS,
                );
            }
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

/// For bosonic rules, ordinary trace and full tensor trace coincide. Fermionic
/// rules are intentionally covered by fixed trace-vs-supertrace oracles.
#[test]
fn bosonic_trace_matches_partial_trace_engine() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            if !$fermionic {
                let v = GradedSpace::try_new($provider, $pairs).unwrap();
                for (ncod, seed) in [(1usize, 71u64), (2, 72)] {
                    let cod: Vec<_> = std::iter::repeat(&v).take(ncod).collect();
                    let pairs: Vec<(usize, usize)> = (0..ncod).map(|i| (i, ncod + i)).collect();
                    let real: TensorMap<_, f64> =
                        TensorMap::rand_with_seed(&rt, cod.clone(), cod.clone(), seed).unwrap();
                    numerics::assert_close(
                        "real trace_pairs",
                        real.tr().unwrap(),
                        real.trace_pairs(&pairs).unwrap().scalar().unwrap(),
                        TERMS,
                    );
                    let complex: TensorMap<_, Complex64> =
                        TensorMap::rand_with_seed(&rt, cod.clone(), cod, seed).unwrap();
                    let fast = complex.tr().unwrap();
                    let engine = complex.trace_pairs(&pairs).unwrap().scalar().unwrap();
                    numerics::assert_close("complex trace_pairs re", fast.re, engine.re, TERMS);
                    numerics::assert_close("complex trace_pairs im", fast.im, engine.im, TERMS);
                }
            }
            let _ = $name;
        }};
    }
    for_each_space!(case);
}

/// `tr(id(V)) = dim(V)` in both dtypes for Abelian, non-Abelian, fermionic,
/// and product rules. This independently pins the positive dimension weight.
#[test]
fn ordinary_trace_of_identity_is_positive_dimension() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            let expected = v.dim().unwrap();
            let real: TensorMap<_, f64> = TensorMap::isomorphism(&rt, [&v], [&v]).unwrap();
            numerics::assert_close("real trace", real.tr().unwrap(), expected, TERMS);
            let complex: TensorMap<_, Complex64> = TensorMap::isomorphism(&rt, [&v], [&v]).unwrap();
            let actual = complex.tr().unwrap();
            numerics::assert_close("complex trace re", actual.re, expected, TERMS);
            numerics::assert_close("complex trace im", actual.im, 0.0, TERMS);
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

/// twist² == id on every leg (all rules in scope have θ ∈ {±1}); bosonic
/// rules have trivial twist; twist is natural with respect to permute.
#[test]
fn twist_squares_to_identity_and_naturality() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0004u64;
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            let t: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 71).unwrap();
            for leg in 0..4usize {
                let twice = t
                    .twist(&[leg], Direction::Forward)
                    .unwrap()
                    .twist(&[leg], Direction::Forward)
                    .unwrap();
                numerics::assert_slices_close(
                    "double twist",
                    twice.dense_data().unwrap(),
                    t.dense_data().unwrap(),
                    TERMS,
                );
                let once = t.twist(&[leg], Direction::Forward).unwrap();
                if $fermionic {
                    // Every leg of these fermionic fixtures carries an odd sector,
                    // so the twist must negate those blocks. Guards the
                    // has_shared_twist short-circuit against wrongly skipping the
                    // sign (θ²=id alone would not catch it — doing nothing also
                    // squares to the identity).
                    assert!(
                        once.dense_data().unwrap() != t.dense_data().unwrap(),
                        "{}: fermionic twist on leg {leg} must not be a no-op",
                        $name,
                    );
                } else {
                    // Bosonic: θ ≡ 1, so the twist is the identity and the
                    // short-circuit returns the shared buffer unchanged.
                    numerics::assert_slices_close(
                        "bosonic twist",
                        once.dense_data().unwrap(),
                        t.dense_data().unwrap(),
                        TERMS,
                    );
                }
            }
            let s = rand_perm(&mut state, 4);
            let pos = s.iter().position(|&j| j == 0).unwrap();
            let lhs = t
                .twist(&[0], Direction::Forward)
                .unwrap()
                .permute(&s[..2], &s[2..])
                .unwrap();
            let rhs = t
                .permute(&s[..2], &s[2..])
                .unwrap()
                .twist(&[pos], Direction::Forward)
                .unwrap();
            numerics::assert_slices_close(
                "identity",
                lhs.dense_data().unwrap(),
                rhs.dense_data().unwrap(),
                TERMS,
            );
        }};
    }
    for_each_space!(case);
}

/// isometry / isomorphism isometric identities: `w†∘w == id`.
#[test]
fn isometry_and_isomorphism_are_isometric() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            let id: TensorMap<_, f64> = TensorMap::isomorphism(&rt, [&v], [&v]).unwrap();
            numerics::assert_slices_close(
                "identity isometry",
                id.adjoint()
                    .unwrap()
                    .compose(&id)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                id.dense_data().unwrap(),
                TERMS,
            );
            let w: TensorMap<_, f64> = TensorMap::isometry(&rt, [&v, &v], [&v]).unwrap();
            numerics::assert_slices_close(
                "isometry",
                w.adjoint()
                    .unwrap()
                    .compose(&w)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                id.dense_data().unwrap(),
                TERMS,
            );
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

/// Norm, trace and singular values agree on one quantity:
/// `tr(e†∘e) = ‖e‖² = Σ_c dim(c) Σ σ_c²`, for a rank-2 ← 2 composition `e`.
#[test]
fn norm_trace_and_singular_values_measure_the_same_weighted_square() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let provider = $provider;
            let v = GradedSpace::try_new(Arc::clone(&provider), $pairs).unwrap();
            let a: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 81).unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 82).unwrap();
            let e = a.compose(&b).unwrap();
            let square = e.norm(2.0).unwrap().powi(2);
            let trace = e.adjoint().unwrap().compose(&e).unwrap().tr().unwrap();
            numerics::assert_close($name, trace, square, TERMS);
            let weighted: f64 = e
                .svd_vals(&codomain_axes(&e), &domain_axes(&e))
                .unwrap()
                .iter()
                .map(|spectrum| {
                    let dim = GradedSpace::try_new(Arc::clone(&provider), [(spectrum.sector, 1)])
                        .unwrap()
                        .dim()
                        .unwrap();
                    dim * spectrum.values.iter().map(|s| s * s).sum::<f64>()
                })
                .sum();
            numerics::assert_close($name, weighted, square, TERMS);
            let _ = $fermionic;
        }};
    }
    for_each_space!(case);
}

// ---------------------------------------------------------------------------
// Contraction order independence
// ---------------------------------------------------------------------------

/// The same network contracted through different explicit pairwise routes
/// must agree, for both a closed ring (scalar) and an open two-tensor network
/// that forces axis permutations.
#[test]
fn contraction_order_independence() {
    let rt = Runtime::builder().build().unwrap();
    macro_rules! case {
        ($name:expr, $provider:expr, $pairs:expr, $fermionic:expr) => {{
            let v = GradedSpace::try_new($provider, $pairs).unwrap();
            // Closed ring of four matrices: full tensor trace of x1 x2 x3 x4.
            let x1: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v], [&v], 81).unwrap();
            let x2: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v], [&v], 82).unwrap();
            let x3: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v], [&v], 83).unwrap();
            let x4: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v], [&v], 84).unwrap();
            let left = x1
                .compose(&x2)
                .unwrap()
                .compose(&x3)
                .unwrap()
                .compose(&x4)
                .unwrap()
                .trace_pairs(&[(0, 1)])
                .unwrap()
                .scalar()
                .unwrap();
            let inner = x2.compose(&x3).unwrap();
            let middle = x1
                .compose(&inner)
                .unwrap()
                .compose(&x4)
                .unwrap()
                .trace_pairs(&[(0, 1)])
                .unwrap()
                .scalar()
                .unwrap();
            numerics::assert_close("trace association", left, middle, TERMS);

            // Open chain: both association orders agree elementwise.
            let assoc_l = x1.compose(&x2).unwrap().compose(&x3).unwrap();
            let assoc_r = x1.compose(&x2.compose(&x3).unwrap()).unwrap();
            numerics::assert_slices_close(
                "composition associativity",
                assoc_l.dense_data().unwrap(),
                assoc_r.dense_data().unwrap(),
                TERMS,
            );

            // Rank-4 pair with crossed contracted legs: forces tree transforms
            // and output permutes on both routes.
            let a: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 85).unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 86).unwrap();
            let ab = a
                .contract(
                    &b,
                    &ContractSpec {
                        lhs: &[1, 2],
                        rhs: &[2, 1],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                )
                .unwrap()
                // default open order [p, s, q, r] with codomain split after 2
                .permute(&[0, 2], &[3, 1])
                .unwrap();
            let ba = b
                .contract(
                    &a,
                    &ContractSpec {
                        lhs: &[2, 1],
                        rhs: &[1, 2],
                        codomain: &[0, 1],
                        domain: &[2, 3],
                    },
                )
                .unwrap()
                // default open order [q, r, p, s]
                .permute(&[2, 0], &[1, 3])
                .unwrap();
            numerics::assert_slices_close(
                "crossed contraction",
                ab.dense_data().unwrap(),
                ba.dense_data().unwrap(),
                TERMS,
            );
            let _ = ($name, $fermionic);
        }};
    }
    for_each_space!(case);
}

/// Regression test for issue #12: SU(2) with non-uniform sector
/// degeneracies (SU(2) sectors `[(0, 2), (1, 2), (2, 1)]`) used to fail every
/// contract route that needs source tree transforms with
/// `Operation(ShapeMismatch { dst: [2, 2, 2, 2], src: [2, 1, 1, 2] })` —
/// `infer_core_dst_shapes` keyed inferred shapes by
/// `(lhs codomain tree, rhs domain tree)`, which is only the destination key
/// when the contracted axes are exactly lhs-domain x rhs-codomain (compose);
/// for crossed axes the open-axis shapes of unrelated sector combinations
/// collided under one key. Asserts contraction-order independence between
/// the two explicit crossed-contract routes.
#[test]
fn su2_nonuniform_degeneracy_crossed_contract() {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(SU2FusionRule);
    let v = GradedSpace::try_new(
        rule,
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap();
    let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 85).unwrap();
    let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 86).unwrap();
    let ab = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[1, 2],
                rhs: &[2, 1],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap()
        .permute(&[0, 2], &[3, 1])
        .unwrap();
    let ba = b
        .contract(
            &a,
            &ContractSpec {
                lhs: &[2, 1],
                rhs: &[1, 2],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap()
        .permute(&[2, 0], &[1, 3])
        .unwrap();
    numerics::assert_slices_close(
        "contraction order",
        ab.dense_data().unwrap(),
        ba.dense_data().unwrap(),
        TERMS,
    );
}

/// Regression test for issue #12, fZ2 shape: decreasing degeneracies
/// (`fZ2` sectors `[(0, 2), (1, 1)]`, the fused norm leg of the finite-torus
/// DL network) with a single contracted leg moving open legs across the
/// codomain/domain boundary. Same root cause as the SU(2) case above; the
/// reference route (permute first, then plain compose) never triggered it.
#[test]
fn fz2_decreasing_degeneracy_boundary_crossing_contract() {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(FermionParityFusionRule);
    let v = GradedSpace::try_new(rule, [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)]).unwrap();
    let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 5).unwrap();
    let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 6).unwrap();
    // Open legs cross the split: a's domain axis 3 stays open, b's axes
    // 1..3 stay open. Default output order matches the permuted compose.
    let direct = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1, 2],
                domain: &[3, 4, 5],
            },
        )
        .unwrap();
    let reference = a
        .permute(&[0, 1, 3], &[2])
        .unwrap()
        .compose(&b.permute(&[0], &[1, 2, 3]).unwrap())
        .unwrap();
    numerics::assert_slices_close(
        "direct contraction",
        direct.dense_data().unwrap(),
        reference.dense_data().unwrap(),
        TERMS,
    );
}

/// Regression test for issue #12, triple-product shape: non-uniform
/// degeneracies on the fZ2 x U1 x SU2 product rule with crossed contracted
/// legs (the original suite only covered the degeneracy-1 triple space).
#[test]
fn triple_product_nonuniform_degeneracy_crossed_contract() {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(Fz2U1Su2Rule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    ));
    let label = |parity, charge, spin| {
        product_sector(
            product_sector(parity, U1Irrep::new(charge)),
            SU2Irrep::from_twice_spin(spin),
        )
    };
    let v = GradedSpace::try_new(
        rule,
        [
            (label(Z2Irrep::EVEN, 0, 0), 2),
            (label(Z2Irrep::ODD, 1, 1), 2),
            (label(Z2Irrep::EVEN, 2, 0), 1),
        ],
    )
    .unwrap();
    let a: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 87).unwrap();
    let b: TensorMap<_, f64> = TensorMap::rand_with_seed(&rt, [&v, &v], [&v, &v], 88).unwrap();
    let ab = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[1, 2],
                rhs: &[2, 1],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap()
        .permute(&[0, 2], &[3, 1])
        .unwrap();
    let ba = b
        .contract(
            &a,
            &ContractSpec {
                lhs: &[2, 1],
                rhs: &[1, 2],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap()
        .permute(&[2, 0], &[1, 3])
        .unwrap();
    numerics::assert_slices_close(
        "contraction order",
        ab.dense_data().unwrap(),
        ba.dense_data().unwrap(),
        TERMS,
    );
}

// ---------------------------------------------------------------------------
// Decompositions over random sector content
// ---------------------------------------------------------------------------

/// svd / qr reconstruction and isometry identities over randomly drawn
/// sector contents (beyond the fixed fixture spaces).
#[test]
fn svd_qr_reconstruction_random_spaces() {
    let rt = Runtime::builder().build().unwrap();
    let mut state = 0x5EED_0005u64;
    macro_rules! case {
        ($name:expr, $provider:expr, $vacuum:expr, [$($other:expr),* $(,)?]) => {{
        let provider = $provider;
        let build = |state: &mut u64| {
            let mut sectors = vec![($vacuum, 1 + rand_below(state, 2))];
            $(
                if rand_below(state, 2) == 1 {
                    sectors.push(($other, 1 + rand_below(state, 2)));
                }
            )*
            GradedSpace::try_new(Arc::clone(&provider), sectors).unwrap()
        };
        for draw in 0..3u64 {
            let va = build(&mut state);
            let vb = build(&mut state);
            let t: TensorMap<_, f64> =
                TensorMap::rand_with_seed(&rt, [&va, &vb], [&vb, &va], 90 + draw).unwrap();
            if t.norm(2.0).unwrap() == 0.0 {
                continue;
            }

            let Svd { u, s, vh } = t.svd_compact(&[0, 1], &[2, 3]).unwrap();
            let recon = u.compose(&s).unwrap().compose(&vh).unwrap();
            let diff = recon.axpby(1.0, &t, -1.0).unwrap().norm(2.0).unwrap();
            assert!(
                diff <= 1e-10 * (1.0 + t.norm(2.0).unwrap()),
                "{} draw {draw}: svd reconstruction error {diff}",
                $name,
            );
            let mid = u.domain();

            let id: TensorMap<_, f64> = TensorMap::isomorphism(&rt, mid.iter(), mid.iter()).unwrap();
            let utu = u.adjoint().unwrap().compose(&u).unwrap();
            let iso_err = utu.axpby(1.0, &id, -1.0).unwrap().norm(2.0).unwrap();
            assert!(
                iso_err <= 1e-10,
                "{} draw {draw}: U†U != id ({iso_err})",
                $name,
            );

            let Qr { q, r } = t.qr_compact(&[0, 1], &[2, 3]).unwrap();
            let recon = q.compose(&r).unwrap();
            let diff = recon.axpby(1.0, &t, -1.0).unwrap().norm(2.0).unwrap();
            assert!(
                diff <= 1e-10 * (1.0 + t.norm(2.0).unwrap()),
                "{} draw {draw}: qr reconstruction error {diff}",
                $name,
            );
            let mid = q.domain();

            let id: TensorMap<_, f64> = TensorMap::isomorphism(&rt, mid.iter(), mid.iter()).unwrap();
            let qtq = q.adjoint().unwrap().compose(&q).unwrap();
            let iso_err = qtq.axpby(1.0, &id, -1.0).unwrap().norm(2.0).unwrap();
            assert!(
                iso_err <= 1e-10,
                "{} draw {draw}: Q†Q != id ({iso_err})",
                $name,
            );
        }
        }};
    }
    case!("Z2", Arc::new(Z2FusionRule), Z2Irrep::EVEN, [Z2Irrep::ODD]);
    case!(
        "fZ2",
        Arc::new(FermionParityFusionRule),
        Z2Irrep::EVEN,
        [Z2Irrep::ODD]
    );
    case!(
        "U1",
        Arc::new(U1FusionRule),
        U1Irrep::new(0),
        [
            U1Irrep::new(-2),
            U1Irrep::new(-1),
            U1Irrep::new(1),
            U1Irrep::new(2),
        ]
    );
    case!(
        "SU2",
        Arc::new(SU2FusionRule),
        SU2Irrep::from_twice_spin(0),
        [SU2Irrep::from_twice_spin(1), SU2Irrep::from_twice_spin(2),]
    );
    let triple = Arc::new(Fz2U1Su2Rule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    ));
    let label = |parity, charge, spin| {
        product_sector(
            product_sector(parity, U1Irrep::new(charge)),
            SU2Irrep::from_twice_spin(spin),
        )
    };
    case!(
        "fZ2xU1xSU2",
        triple,
        label(Z2Irrep::EVEN, 0, 0),
        [
            label(Z2Irrep::ODD, 1, 1),
            label(Z2Irrep::EVEN, 2, 0),
            label(Z2Irrep::ODD, -1, 1),
        ]
    );
}

// ---------------------------------------------------------------------------
// Part 3: weighted rank truncation vs TensorKit
// ---------------------------------------------------------------------------

/// The shared deterministic per-block fill, the same function the TensorKit
/// 0.17.1 run used to produce the truncation values below. Labels: U1 charge
/// or SU2 twice-spin; indices are one-based degeneracy coordinates, codomain
/// axes first.
fn oracle_fill(c0: i64, labels: [i64; 5], idx: &[usize]) -> f64 {
    let [l1, l2, m1, m2, lc] = labels;
    let weighted = c0
        + 7 * l1
        + 11 * l2
        + 13 * m1
        + 17 * m2
        + 19 * lc
        + 23 * (idx[0] as i64 + 1)
        + 29 * (idx[1] as i64 + 1)
        + 31 * (idx[2] as i64 + 1)
        + 37 * (idx[3] as i64 + 1);
    (weighted.rem_euclid(41) - 20) as f64
}

/// TensorKit `truncrank(5)` counts quantum dimensions, not just the number of
/// stored singular values. The SU(2) fixture therefore keeps one spin-1/2
/// value (weight 2) and one spin-1 value (weight 3), while U(1) keeps five
/// scalar-weight values.
#[test]
fn weighted_rank_truncation_matches_tensorkit() {
    macro_rules! case {
        ($provider:expr, $pairs:expr, $label_of:expr, $expected:expr, $error:expr) => {{
            let runtime = Runtime::builder().build().unwrap();
            let space = GradedSpace::try_new($provider, $pairs).unwrap();
            let label_of = $label_of;
            let build = |c0| {
                TensorMap::from_subblock_fn(
                    &runtime,
                    [&space, &space],
                    [&space, &space],
                    |trees, idx| {
                        let codomain = trees.codomain_uncoupled();
                        let domain = trees.domain_uncoupled();
                        oracle_fill(
                            c0,
                            [
                                label_of(&codomain[0]),
                                label_of(&codomain[1]),
                                label_of(&domain[0]),
                                label_of(&domain[1]),
                                label_of(trees.coupled()),
                            ],
                            idx,
                        )
                    },
                )
                .unwrap()
            };
            let a = build(3);
            let b = build(5);
            let e = a
                .permute(&[1, 0], &[3, 2])
                .unwrap()
                .compose(&a.compose(&b).unwrap())
                .unwrap();
            let Svd { s, .. } = e.svd_compact(&[0, 1], &[2, 3]).unwrap();
            let found = s.domain()[0]
                .find_truncated(&s.diagview().unwrap(), &Truncation::rank(5))
                .unwrap();
            // `diagview` lists sectors in stored order; compare by label.
            let mut kept: Vec<_> = s
                .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
                .unwrap()
                .diagview()
                .unwrap()
                .iter()
                .filter(|entry| !entry.values.is_empty())
                .map(|entry| (label_of(&entry.sector), entry.values.len()))
                .collect();
            kept.sort_unstable();
            assert_eq!(kept, $expected);
            numerics::assert_close_scaled(
                "truncation error",
                found.error,
                $error,
                1,
                envelope(1e-10),
            );
        }};
    }

    case!(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
        |sector: &U1Irrep| sector.charge() as i64,
        [(-1, 1), (0, 2), (1, 2)],
        155706.08009324488
    );
    case!(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
        |sector: &SU2Irrep| sector.twice_spin() as i64,
        [(1, 1), (2, 1)],
        276850.04205868527
    );
}
