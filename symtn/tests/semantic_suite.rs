//! Operation-identity property tests over the user layer (issue #9, part 2)
//! plus the cross-library invariant-stream check against TensorKit (part 3).
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
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule,
    SU2Irrep, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
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
type Fz2U1Sector = ProductSector<Z2Irrep, U1Irrep>;
type Fz2U1Su2Sector = ProductSector<Fz2U1Sector, SU2Irrep>;

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
// Part 3: cross-library invariant stream vs TensorKit
// ---------------------------------------------------------------------------

/// The shared deterministic per-block fill, the same function the TensorKit
/// 0.17.1 run used to produce the expected streams below. Labels: U1 charge
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

/// Runs the part-3 op sequence and compares the basis-independent
/// invariants (norm, tr, singular values) against the committed TensorKit
/// stream. `svd_expected` holds the significant singular values (>= 1e-6
/// of the largest); trailing numerically-zero values are only counted.
macro_rules! invariant_stream_case {
    ($provider:expr, $pairs:expr, $label_of:expr, $expected:expr, $svd_count:expr, $svd_expected:expr) => {{
        let rt = Runtime::builder().build().unwrap();
        let v = GradedSpace::try_new($provider, $pairs).unwrap();
        let label_of = $label_of;
        let build = |c0| {
            TensorMap::from_subblock_fn(&rt, [&v, &v], [&v, &v], |trees, idx| {
                let cod = trees.codomain_uncoupled();
                let dom = trees.domain_uncoupled();
                oracle_fill(
                    c0,
                    [
                        label_of(&cod[0]),
                        label_of(&cod[1]),
                        label_of(&dom[0]),
                        label_of(&dom[1]),
                        label_of(trees.coupled()),
                    ],
                    idx,
                )
            })
            .unwrap()
        };
        let a = build(3);
        let b = build(5);
        let c = a.compose(&b).unwrap();
        let d = a.permute(&[1, 0], &[3, 2]).unwrap();
        let e = d.compose(&c).unwrap();
        let g = a.adjoint().unwrap().compose(&a).unwrap();
        let h = e.axpby(1.0, &a, 0.5).unwrap();
        let hh_tr = h.compose(&h).unwrap().tr().unwrap();

        let steps = [
            ("s1a", a.norm(2.0).unwrap(), a.tr().unwrap()),
            ("s1b", b.norm(2.0).unwrap(), b.tr().unwrap()),
            ("s2", c.norm(2.0).unwrap(), c.tr().unwrap()),
            ("s3", d.norm(2.0).unwrap(), d.tr().unwrap()),
            ("s4", e.norm(2.0).unwrap(), e.tr().unwrap()),
            ("s5", g.norm(2.0).unwrap(), g.tr().unwrap()),
            ("s7", h.norm(2.0).unwrap(), h.tr().unwrap()),
            ("s8", hh_tr, hh_tr),
        ];
        for ((step, norm, tr), &(exp_step, exp_norm, exp_tr)) in steps.iter().zip($expected) {
            assert_eq!(*step, exp_step);
            numerics::assert_close_scaled("stream norm", *norm, exp_norm, 1, envelope(1e-9));
            numerics::assert_close_scaled("stream trace", *tr, exp_tr, 1, envelope(1e-9));
        }

        let mut values: Vec<f64> = e
            .svd_vals(&codomain_axes(&e), &domain_axes(&e))
            .unwrap()
            .iter()
            .flat_map(|spectrum| spectrum.values.iter().copied())
            .collect();
        values.sort_by(|x, y| y.partial_cmp(x).unwrap());
        assert_eq!(values.len(), $svd_count, "singular value count");
        let cutoff = 1e-6 * $svd_expected[0];
        for (k, (&got, &exp)) in values.iter().zip($svd_expected).enumerate() {
            numerics::assert_close_scaled("stream singular value", got, exp, 1, envelope(1e-8));
            assert!(exp > cutoff, "svd_expected[{k}] below cutoff");
        }
        for &tail in &values[$svd_expected.len()..] {
            assert!(
                tail <= cutoff,
                "unexpected significant singular value {tail}"
            );
        }
    }};
}

/// Cross-library invariant stream, U(1). Expected norms, traces and singular
/// values checked against TensorKit 0.17.1 (Julia 1.11.6).
/// Run with: `cargo test -p symtn --release --test semantic_suite -- --ignored u1_vs`
#[test]
#[ignore = "cross-library stream, run explicitly (release recommended)"]
fn cross_library_invariant_stream_u1_vs_tensorkit() {
    let expected = [
        ("s1a", 2.887628785006827e2, -1.9e1),
        ("s1b", 2.896515147552315e2, -3.0),
        ("s2", 1.402426411616665e4, 1.104e4),
        ("s3", 2.887628785006826e2, -1.9e1),
        ("s4", 8.209534152002786e5, 1.03394e5),
        ("s5", 2.485837810477585e4, 8.3384e4),
        ("s7", 8.209824186503642e5, 1.033845e5),
        ("s8", 6.256589682425e10, 6.256589682425e10),
    ];
    let svd = [
        5.938452457943761e5,
        4.179528114697005e5,
        2.459326023786228e5,
        1.933057950606111e5,
        1.566314868289549e5,
        8.210318973109889e4,
        7.523003903903847e4,
        6.565202259389844e4,
        5.151867957441879e4,
        3.767419629678344e4,
        2.538188406982896e4,
        2.343_136_052_737_89e4,
        2.008745263879226e4,
        1.964054032660374e4,
        1.883995230847887e4,
        1.867342767080404e4,
        1.396252249681011e4,
        1.278185908520561e4,
        1.221617279960168e4,
        9.646640464331718e3,
        7.594820923703089e3,
        7.588096644934964e3,
        6.218_196_863_168_91e3,
        2.651408037571802e3,
        2.163357342259171e3,
        1.657_615_939_186_76e3,
        1.643314514396656e3,
        1.129212243761851e3,
        6.562754711025852e2,
        5.535225377189289e2,
        5.476578287137901e2,
    ];
    invariant_stream_case!(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
        |sector: &U1Irrep| i64::from(sector.charge()),
        &expected,
        49,
        &svd
    );
}

/// Cross-library invariant stream, SU(2). Expected norms, traces and singular
/// values checked against TensorKit 0.17.1 (Julia 1.11.6).
/// Run with: `cargo test -p symtn --release --test semantic_suite -- --ignored su2_vs`
#[test]
#[ignore = "cross-library stream, run explicitly (release recommended)"]
fn cross_library_invariant_stream_su2_vs_tensorkit() {
    let expected = [
        ("s1a", 3.075792580783041e2, -2.24e2),
        ("s1b", 3.089012787283341e2, -1.44e2),
        ("s2", 1.423541506244198e4, 1.6685e4),
        ("s3", 3.075792580783042e2, -2.24e2),
        ("s4", 7.171108525311829e5, -1.481890000000002e5),
        ("s5", 2.043484428616964e4, 9.4605e4),
        ("s7", 7.171433641554874e5, -1.483010000000002e5),
        ("s8", 2.412084471050002e10, 2.412084471050002e10),
    ];
    let svd = [
        4.110_592_957_633_28e5,
        1.822658678626898e5,
        1.406505246682399e5,
        1.140_569_967_151_56e5,
        7.343460378313088e4,
        5.224807565545758e4,
        4.043006489580932e4,
        2.351544167506671e4,
        2.184633501094873e4,
        1.928046828460497e4,
        1.660174629027798e4,
        8.566028701881454e3,
        6.373677277671596e3,
        4.039539983749055e3,
        3.958321987177687e3,
        3.574358904429296e3,
        3.288748484860358e3,
        3.151829069086835e3,
        1.779769959383261e3,
        1.439999999999999e3,
        1.076447526243916e3,
        5.845450724371532e2,
    ];
    invariant_stream_case!(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
        |sector: &SU2Irrep| i64::try_from(sector.twice_spin()).unwrap(),
        &expected,
        35,
        &svd
    );
}

/// Cross-library invariant stream for fZ2 x U(1) x SU(2). Expected values
/// checked against TensorKit 0.17.1 (Julia 1.11.6).
/// Run with: `cargo test -p symtn --release --test semantic_suite -- --ignored product_vs`
#[test]
#[ignore = "cross-library stream, run explicitly (release recommended)"]
fn cross_library_invariant_stream_product_vs_tensorkit() {
    let expected = [
        ("s1a", 2.041910869749216e2, -8.0),
        ("s1b", 2.060849339471471e2, 4.9e1),
        ("s2", 7.029845233004778e3, 1.269e4),
        ("s3", 2.041910869749217e2, -8.000000000000014),
        ("s4", 2.048_990_970_819_54e5, 1.251150000000001e5),
        ("s5", 1.041952436534413e4, 4.1694e4),
        ("s7", 2.049323716485515e5, 1.251110000000001e5),
        ("s8", 1.392459691750001e10, 1.392459691750001e10),
    ];
    let svd = [
        9.313322524156692e4,
        8.377972168698783e4,
        5.801978514614065e4,
        5.425702935363993e4,
        3.244521596397725e4,
        2.349_078_948_391_45e4,
        2.264344851117111e4,
        2.189538877119227e4,
        2.159365106169731e4,
        1.869173451469769e4,
        1.486434829405537e4,
        7.22e3,
        5.912689186545615e3,
        5.486716759584609e3,
        4.241123285758052e3,
        1.977685541462147e3,
        7.714938309221702e2,
        7.163293805552347e2,
        6.015165827414188e2,
        5.900887872343542e2,
        2.167139622628681e2,
    ];
    let provider = Arc::new(Fz2U1Su2Rule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    ));
    invariant_stream_case!(
        provider,
        [
            (
                product_sector(
                    product_sector(Z2Irrep::EVEN, U1Irrep::new(0)),
                    SU2Irrep::from_twice_spin(0),
                ),
                2,
            ),
            (
                product_sector(
                    product_sector(Z2Irrep::ODD, U1Irrep::new(1)),
                    SU2Irrep::from_twice_spin(1),
                ),
                2,
            ),
            (
                product_sector(
                    product_sector(Z2Irrep::EVEN, U1Irrep::new(2)),
                    SU2Irrep::from_twice_spin(0),
                ),
                1,
            ),
        ],
        |sector: &Fz2U1Su2Sector| {
            100 * i64::from(sector.left().left().parity())
                + 10 * i64::from(sector.left().right().charge())
                + i64::try_from(sector.right().twice_spin()).unwrap()
        },
        &expected,
        29,
        &svd
    );
}
