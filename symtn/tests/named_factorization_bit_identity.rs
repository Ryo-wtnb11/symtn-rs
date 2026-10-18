//! #1541: the named factorization results carry exactly the factors the tuple
//! API returned. Each result is destructured back into the old tuple order and
//! fingerprinted twice. The structural fingerprint (legs, block order, storage
//! class, payload length) is portable and checked everywhere. The exact
//! fingerprint adds every materialized value in round-trip `Debug` form, which
//! identifies each `f64` bit pattern; since the dense kernels pick SIMD paths
//! by CPU, its expected values hold only for the recording host class (macOS
//! aarch64). The original expected values were recorded with the tuple API on
//! `origin/main` a2ce6c49, single-threaded dense executor, `cpu-faer`.
//! The SU(3) compact SVD storage hashes reflect compact `s`; its
//! materialized factor values are checked against the original exact hashes.
//! The SU(3) lazy-adjoint SVD, QR and LQ rows were recorded at #1755.
//! The multiplicity-free polar rows (direct and lazy adjoint) were re-recorded
//! when polar moved to the shared per-sector MatrixAlgebraKit `PolarViaSVD`
//! kernel (#1752): structure unchanged, values within rounding.
//! The `eigh_full`/`eig_full` exact rows (direct and lazy adjoint) were
//! re-recorded when eigenvalues became ascending (#1985): structure
//! unchanged, values and columns permuted within each sector.
//! The `eig_full` exact rows (direct and lazy adjoint) were re-recorded when
//! eigenvector columns became unit 2-norm (#2073): structure unchanged.

#[path = "../../tests/support/backend.rs"]
mod backend;

use num_complex::Complex64;
use std::sync::Arc;
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::__network::NetworkReuseClass;
use symtn::typed::HermitianTol;
use symtn::typed::{Eig, Eigh, GradedSpace, LeftPolar, Lq, Qr, RightPolar, Svd, TensorMap};

// The old tuple order of every result.
fn svd<T>(x: Svd<T>) -> (T, T, T) {
    (x.u, x.s, x.vh)
}
fn qr<T>(x: Qr<T>) -> (T, T) {
    (x.q, x.r)
}
fn lq<T>(x: Lq<T>) -> (T, T) {
    (x.l, x.q)
}
fn eigh<T>(x: Eigh<T>) -> (T, T) {
    (x.d, x.v)
}
fn eig<T>(x: Eig<T>) -> (T, T) {
    (x.d, x.v)
}
fn left_polar<T>(x: LeftPolar<T>) -> (T, T) {
    (x.w, x.p)
}
fn right_polar<T>(x: RightPolar<T>) -> (T, T) {
    (x.p, x.wh)
}

fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

macro_rules! factor {
    ($t:expr) => {{
        let t = $t;
        let legs = |legs: Vec<GradedSpace<_>>| {
            legs.iter()
                .map(|leg| {
                    let sectors = leg.sectors().unwrap();
                    format!("{sectors:?}{:?}{}", leg.degeneracies(), leg.is_dual())
                })
                .collect::<Vec<_>>()
        };
        let blocks = (0..t.subblock_count())
            .map(|i| format!("{:?}", t.subblock_fusion_trees(i).unwrap().coupled()))
            .collect::<Vec<_>>();
        let (codomain, domain) = (legs(t.codomain()), legs(t.domain()));
        let compact = matches!(
            symtn::typed::__network::network_reuse_class(&t, false),
            NetworkReuseClass::Compact
        );
        let data_materialized = t.materialize().unwrap();
        let data = data_materialized.dense_data().unwrap();
        (
            format!(
                "{codomain:?}<-{domain:?}|{blocks:?}|compact={compact}|len={}",
                data.len()
            ),
            format!("{codomain:?}<-{domain:?}|{blocks:?}|compact={compact}|{data:?}"),
        )
    }};
}

macro_rules! factors {
    ($result:expr, $split:expr, $($field:tt),+) => {
        match $result {
            Ok(value) => {
                let tuple = $split(value);
                let parts = [$(factor!(&tuple.$field)),+];
                (
                    parts.iter().map(|part| part.0.as_str()).collect::<Vec<_>>().join(";"),
                    parts.iter().map(|part| part.1.as_str()).collect::<Vec<_>>().join(";"),
                )
            }
            Err(_) => ("Err".to_string(), "Err".to_string()),
        }
    };
}

/// Fingerprints of every factorization, lazy-adjoint inputs included, of the
/// fixture `tall : [v, v] <- [v]`, `wide = tall^H`-shaped, a square
/// endomorphism, and a Hermitian endomorphism.
macro_rules! fingerprints {
    ($runtime:expr, $v:expr, $d:ty, $seed:expr) => {{
        let (runtime, v, seed) = ($runtime, $v, $seed);
        let tall = TensorMap::<_, $d>::rand_with_seed(runtime, [v, v], [v], seed).unwrap();
        let wide = TensorMap::<_, $d>::rand_with_seed(runtime, [v], [v, v], seed + 1).unwrap();
        let endo = TensorMap::<_, $d>::rand_with_seed(runtime, [v], [v], seed + 2).unwrap();
        // Real symmetric blocks with distinct diagonal: Hermitian for both
        // dtypes, built without a contraction (checked Generic rejects lazy
        // operands there).
        let herm = TensorMap::<_, $d>::from_subblock_fn(runtime, [v], [v], |_, ij| {
            let (i, j) = (ij[0] as f64, ij[1] as f64);
            <$d>::from(
                1.0 / (1.0 + i + j)
                    + if ij[0] == ij[1] {
                        0.37 * (1.0 + i)
                    } else {
                        0.0
                    },
            )
        })
        .unwrap();
        let (tall_h, wide_h) = (tall.adjoint().unwrap(), wide.adjoint().unwrap());
        let (endo_h, herm_h) = (endo.adjoint().unwrap(), herm.adjoint().unwrap());
        [
            factors!(tall.svd_compact(&[0, 1], &[2]), svd, 0, 1, 2),
            factors!(tall.svd_full(&[0, 1], &[2]), svd, 0, 1, 2),
            factors!(tall.qr_compact(&[0, 1], &[2]), qr, 0, 1),
            factors!(wide.qr_full(&[0], &[1, 2]), qr, 0, 1),
            factors!(wide.lq_compact(&[0], &[1, 2]), lq, 0, 1),
            factors!(tall.lq_full(&[0, 1], &[2]), lq, 0, 1),
            factors!(tall.left_polar(&[0, 1], &[2]), left_polar, 0, 1),
            factors!(wide.right_polar(&[0], &[1, 2]), right_polar, 0, 1),
            factors!(
                herm.eigh_full(&[0], &[1], HermitianTol::DEFAULT),
                eigh,
                0,
                1
            ),
            factors!(endo.eig_full(&[0], &[1]), eig, 0, 1),
            factors!(wide_h.svd_compact(&[0, 1], &[2]), svd, 0, 1, 2),
            factors!(tall_h.svd_full(&[0], &[1, 2]), svd, 0, 1, 2),
            factors!(wide_h.qr_compact(&[0, 1], &[2]), qr, 0, 1),
            factors!(tall_h.qr_full(&[0], &[1, 2]), qr, 0, 1),
            factors!(tall_h.lq_compact(&[0], &[1, 2]), lq, 0, 1),
            factors!(wide_h.lq_full(&[0, 1], &[2]), lq, 0, 1),
            factors!(wide_h.left_polar(&[0, 1], &[2]), left_polar, 0, 1),
            factors!(tall_h.right_polar(&[0], &[1, 2]), right_polar, 0, 1),
            factors!(
                herm_h.eigh_full(&[0], &[1], HermitianTol::DEFAULT),
                eigh,
                0,
                1
            ),
            factors!(endo_h.eig_full(&[0], &[1]), eig, 0, 1),
        ]
        .iter()
        .map(|(structure, exact)| (fnv1a(structure), fnv1a(exact)))
        .collect::<Vec<_>>()
    }};
}

const NAMES: [&str; 20] = [
    "svd_compact",
    "svd_full",
    "qr_compact",
    "qr_full",
    "lq_compact",
    "lq_full",
    "left_polar",
    "right_polar",
    "eigh_full",
    "eig_full",
    "adjoint svd_compact",
    "adjoint svd_full",
    "adjoint qr_compact",
    "adjoint qr_full",
    "adjoint lq_compact",
    "adjoint lq_full",
    "adjoint left_polar",
    "adjoint right_polar",
    "adjoint eigh_full",
    "adjoint eig_full",
];

fn check(label: &str, actual: &[(u64, u64)], structure: &[u64; 20], exact: &[u64; 20]) {
    let report = |pick: fn(&(u64, u64)) -> u64| {
        actual
            .iter()
            .map(|hashes| format!("{:#018x},", pick(hashes)))
            .collect::<Vec<_>>()
            .join(" ")
    };
    eprintln!("{label} structure: [{}]", report(|hashes| hashes.0));
    eprintln!("{label} exact: [{}]", report(|hashes| hashes.1));
    for ((name, actual), expected) in NAMES.iter().zip(actual).zip(structure) {
        assert_eq!(actual.0, *expected, "{label} {name} structure");
    }
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) && backend::faer_is_default_backend()
    {
        for ((name, actual), expected) in NAMES.iter().zip(actual).zip(exact) {
            assert_eq!(actual.1, *expected, "{label} {name} exact");
        }
    }
}

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::host_runtime;

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(-1, 1), (0, 2), (1, 2)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(0, 2), (1, 2), (2, 1)].map(|(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
    )
    .unwrap()
}

fn fz2_u1_leg() -> GradedSpace<ProductFusionRule<FermionParityFusionRule, U1FusionRule>> {
    let rule = FermionParityFusionRule.product(U1FusionRule);
    GradedSpace::try_new(
        Arc::new(rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(-1)), 1),
        ],
    )
    .unwrap()
}

/// One test, fixtures in a fixed order: sector ids are interned per process,
/// and block order follows them, so the fingerprints assume this sequence.
#[test]
fn named_results_match_the_tuple_results() {
    let runtime = host_runtime();
    let v = u1_leg();
    check(
        "u1 f64",
        &fingerprints!(&runtime, &v, f64, 1541),
        &U1_F64_STRUCTURE,
        &U1_F64,
    );
    check(
        "u1 c64",
        &fingerprints!(&runtime, &v, Complex64, 1541),
        &U1_C64_STRUCTURE,
        &U1_C64,
    );
    let v = su2_leg();
    check(
        "su2 f64",
        &fingerprints!(&runtime, &v, f64, 1541),
        &SU2_F64_STRUCTURE,
        &SU2_F64,
    );
    check(
        "su2 c64",
        &fingerprints!(&runtime, &v, Complex64, 1541),
        &SU2_C64_STRUCTURE,
        &SU2_C64,
    );
    let v = fz2_u1_leg();
    check(
        "fz2u1 f64",
        &fingerprints!(&runtime, &v, f64, 1541),
        &FZ2U1_F64_STRUCTURE,
        &FZ2U1_F64,
    );
    check(
        "fz2u1 c64",
        &fingerprints!(&runtime, &v, Complex64, 1541),
        &FZ2U1_C64_STRUCTURE,
        &FZ2U1_C64,
    );
    #[cfg(feature = "cgc-gen")]
    {
        // The lazy-adjoint rows were `Err` until checked Generic accepted
        // lazy adjoints (#1755).
        use symtn::sector::SUNFusionRule;
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let v = GradedSpace::try_new(
            provider,
            [(vec![0i64, 0], 2), (vec![1, 0], 2), (vec![0, 1], 1)],
        )
        .unwrap();
        // The compact S changes only the storage marker in the old tuple
        // fingerprint. Check the materialized factors against the old hashes
        // before recording the new representation below.
        macro_rules! unchanged_svd_payload {
            ($dtype:ty, $old_structure:expr, $old_exact:expr) => {{
                let tall =
                    TensorMap::<_, $dtype>::rand_with_seed(&runtime, [&v, &v], [&v], 1541).unwrap();
                let (structure, exact) = factors!(tall.svd_compact(&[0, 1], &[2]), svd, 0, 1, 2);
                assert_eq!(structure.matches("compact=true").count(), 1);
                assert_eq!(exact.matches("compact=true").count(), 1);
                assert_eq!(
                    fnv1a(&structure.replace("compact=true", "compact=false")),
                    $old_structure
                );
                if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
                    assert_eq!(
                        fnv1a(&exact.replace("compact=true", "compact=false")),
                        $old_exact
                    );
                }
            }};
        }
        unchanged_svd_payload!(f64, 0x12cb72c30df11d0f, 0xaa8c6dee3e7d5d6c);
        unchanged_svd_payload!(Complex64, 0x12cb72c30df11d0f, 0x2a24b0233aa370aa);
        check(
            "su3 f64",
            &fingerprints!(&runtime, &v, f64, 1541),
            &SU3_F64_STRUCTURE,
            &SU3_F64,
        );
        check(
            "su3 c64",
            &fingerprints!(&runtime, &v, Complex64, 1541),
            &SU3_C64_STRUCTURE,
            &SU3_C64,
        );
    }
}

const U1_F64_STRUCTURE: [u64; 20] = [
    0xd12f8de2c5e19768,
    0xef9895eddc9720ef,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
    0xd12f8de2c5e19768,
    0x0fcd5f2aae0adc2b,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
];
const U1_F64: [u64; 20] = [
    0xc7d4a9386959fedf,
    0x6ccb371813d66465,
    0x58073f2ddde11363,
    0x088764e125943a52,
    0xe0d6fe066fe182ad,
    0x370c9d660175bbe6,
    0xdd7812350a3f4ec8,
    0x82de2828bec8932a,
    0x01c8ca3cdc7a3b83,
    0x91f3af4dbbec970b,
    0x0ad00b9efda3ea23,
    0xc831e49eeea7726a,
    0xe21e5f8b0b12cffd,
    0x22549abc7bd67c18,
    0x1195baa19ea488ab,
    0xbf80b20bedb48200,
    0x4193b71bd736fe42,
    0xbeec4a45cde9b7e2,
    0x01c8ca3cdc7a3b83,
    0x1c159b8398d2efd7,
];
const U1_C64_STRUCTURE: [u64; 20] = [
    0xd12f8de2c5e19768,
    0xef9895eddc9720ef,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
    0xd12f8de2c5e19768,
    0x0fcd5f2aae0adc2b,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xbf4811b50fc3e100,
    0x03ba48256d1daefc,
    0x03ba48256d1daefc,
    0xbf4811b50fc3e100,
    0xf1a3f63ee1bcf6d9,
    0xf1a3f63ee1bcf6d9,
];
const U1_C64: [u64; 20] = [
    0x927458fc26ab6326,
    0x1e4c19fac7bfa9a8,
    0xe7f79c639e365d95,
    0xee1f76ddd74b8532,
    0xa3a2255a0eab759e,
    0xfa0e10a816a652c2,
    0xa665a8b3830d20b6,
    0xa3d5718b1e79ddd7,
    0xfc608dc675aea21f,
    0xc47fde1c1e55935c,
    0x820e16417a86d790,
    0x9a9746665dabc567,
    0x4c72e12a12e3744f,
    0xbe5cbd5d3fd12ca5,
    0xf1f84c7abdf96f5e,
    0xf375144f69a913fb,
    0xe984b22617206a21,
    0x2066150499b2ccc2,
    0xfc608dc675aea21f,
    0xc11d7d3f768338d8,
];
const SU2_F64_STRUCTURE: [u64; 20] = [
    0xc2735e92fe91606d,
    0x996be5ec295e05dc,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
    0xc2735e92fe91606d,
    0x5daae6758d3203b6,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
];
const SU2_F64: [u64; 20] = [
    0x2c8ead24f3d9d3dc,
    0x058a1f2ccec79691,
    0x4aa8203fdfd9b6e2,
    0xd3d3339297a20af0,
    0xcda57eb3985bbc58,
    0xec6248a2f4a04da5,
    0x2a906745aab0d71f,
    0x7815f711e939d5e2,
    0xcc63c8c0a58077c3,
    0x706e9f197c38b152,
    0x7df281dbbdbeddcc,
    0x57f91b8203aa3551,
    0x9d2cf6c8d254834e,
    0x4400f2902da9b76d,
    0xdf7e90da5b1b6b5c,
    0x0b91da8b7f65cf46,
    0xa99ea8cda147243c,
    0xa113f7c0313cec7b,
    0xcc63c8c0a58077c3,
    0x99dd932c064a51fd,
];
const SU2_C64_STRUCTURE: [u64; 20] = [
    0xc2735e92fe91606d,
    0x996be5ec295e05dc,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
    0xc2735e92fe91606d,
    0x5daae6758d3203b6,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xa7f37e1b09efce8c,
    0xeaf431f215b94494,
    0xeaf431f215b94494,
    0xa7f37e1b09efce8c,
    0xebc4640291babb6b,
    0xebc4640291babb6b,
];
const SU2_C64: [u64; 20] = [
    0x39b2e52328588e9d,
    0x289db2d5e4627c4e,
    0x0e1be73ca1cb1367,
    0x2add883d2170dd50,
    0x465743ea98509911,
    0xc9ba0ed39a9500f3,
    0x34380448a178cfe2,
    0xf6bcec3540540d67,
    0xf324e4cd67bee267,
    0xd87c43ca1fbe3a36,
    0x2e96bbde6011518b,
    0x0bb7cdb54edce565,
    0x4b711d0de531435d,
    0x5299717ad9ada771,
    0x5e4c714d4be4874d,
    0xb18b974b92344566,
    0xf2e6e64a0f7ea396,
    0x4da6bacd4a723b67,
    0xf324e4cd67bee267,
    0x6841c58d6f3360fe,
];
const FZ2U1_F64_STRUCTURE: [u64; 20] = [
    0x0f9538ca0ba4c87a,
    0x04c6b7b5df099c9b,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
    0x0f9538ca0ba4c87a,
    0x5179d2ca9d9ee5cd,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
];
const FZ2U1_F64: [u64; 20] = [
    0x0caf7931bfd3c4cb,
    0xdca9c51a102f39db,
    0x31af1f29f74bd6ec,
    0x8c2abb9739eb28f5,
    0x5cdd525ac34f4390,
    0xcb5c92645fbc3187,
    0xe39ad4999252dfd3,
    0xc7912da1f5f89dfb,
    0xe958a76be06a41ef,
    0x8e91d14f9ec85f47,
    0xde7d102cdbc2a14f,
    0x8bc4f55f46d70698,
    0xb6ac73892efb1c56,
    0xf6d4523afd2f09f7,
    0xe31b1faba2c73af2,
    0xa366fac7915221b9,
    0x3f9d0caa922dfdf1,
    0x062129823edf7ccb,
    0xe958a76be06a41ef,
    0xca816525a8501e1b,
];
const FZ2U1_C64_STRUCTURE: [u64; 20] = [
    0x0f9538ca0ba4c87a,
    0x04c6b7b5df099c9b,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
    0x0f9538ca0ba4c87a,
    0x5179d2ca9d9ee5cd,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xbb366cbdac54d727,
    0xef1a0aa797dc43bb,
    0xef1a0aa797dc43bb,
    0xbb366cbdac54d727,
    0xf753ce7a5ae968bb,
    0xf753ce7a5ae968bb,
];
const FZ2U1_C64: [u64; 20] = [
    0x35b2149eb72b3b96,
    0x3d449c42d4f9e96c,
    0xb8ca7f09b7ec7296,
    0x68b0c48f92ae94a7,
    0x7e714b9c7e42f52b,
    0x597c31fbb4171f8d,
    0xb7f8b14620b066fb,
    0x26eac1f7c791210c,
    0x4cd5a7c2087fdc5d,
    0x0826c9c7fee6be1a,
    0x8d4987d64c522618,
    0x0752768af2b6c871,
    0xd431d41996f08c78,
    0xdb92a7f694a66f0e,
    0xc549648841791991,
    0xc87f5382ec5a540a,
    0x7c887065cb081434,
    0x39422fd9bbf40a9f,
    0x4cd5a7c2087fdc5d,
    0x079c5e9fd6a08602,
];
#[cfg(feature = "cgc-gen")]
const SU3_F64_STRUCTURE: [u64; 20] = [
    0xbc495a3228b0b522,
    0x8cc387613de843c4,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0xf065848bb8da7bd1,
    0x66fd3d88cf5e7459,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
    0xbc495a3228b0b522,
    0x487f7aa30e83f268,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0xf065848bb8da7bd1,
    0x66fd3d88cf5e7459,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
];
#[cfg(feature = "cgc-gen")]
const SU3_F64: [u64; 20] = [
    0x443844ebdff05a81,
    0x2dd2146f4d494ed3,
    0x941829a514907395,
    0x07ec7dd204814640,
    0x833f15b01063ccb4,
    0xd105725ac19f8b11,
    0xe4d5c778a5724c2e,
    0x522d7560a3f5797e,
    0xc6a21cbc83be0d8f,
    0x616a28e3d451f823,
    0x8560b56ad9ad4841,
    0xf62a7bf22ecce6e0,
    0x1cba761a682c44c8,
    0x4e6a043beb5c56b9,
    0x7e6524c1efc4e6a9,
    0x0403981e078d4d34,
    0x1146fc2e2d3e9554,
    0x25ea335e3fa83b42,
    0xc6a21cbc83be0d8f,
    0x34356c817af9385f,
];
#[cfg(feature = "cgc-gen")]
const SU3_C64_STRUCTURE: [u64; 20] = [
    0xbc495a3228b0b522,
    0x8cc387613de843c4,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0xf065848bb8da7bd1,
    0x66fd3d88cf5e7459,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
    0xbc495a3228b0b522,
    0x487f7aa30e83f268,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0xf065848bb8da7bd1,
    0x66fd3d88cf5e7459,
    0x66fd3d88cf5e7459,
    0xf065848bb8da7bd1,
    0x907f9ed0987ba813,
    0x907f9ed0987ba813,
];
#[cfg(feature = "cgc-gen")]
const SU3_C64: [u64; 20] = [
    0x4ac8bb9aba11c1e9,
    0x6ce7c0fa70bc3f2a,
    0xc28cb0b2fc8ad53f,
    0x6f7dc8d068450aef,
    0x0b1fd04660a12253,
    0xa0dfb70e20f6d612,
    0x5e089b61c49000ca,
    0xae97c3bd61d46d67,
    0xd3ec92264f469dc5,
    0xae1bb065a63c7bfa,
    0xa4bcebb91fad459b,
    0xb1fcb918424c6c69,
    0xbae28efae9230d78,
    0x0e7b6742bb17dec1,
    0xc72249df43383f10,
    0x800aaeadaf01f4ce,
    0xd116c41479a9f973,
    0x4af583bb9acb5178,
    0xd3ec92264f469dc5,
    0xa12d4a780089e8f2,
];
