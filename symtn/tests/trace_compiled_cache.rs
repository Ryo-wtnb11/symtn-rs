//! The compiled-trace entry of the completed-transformer cache.
//!
//! A warm fusion-aware trace over canonical structures resolves its whole
//! compiled structure with one completed-transformer lookup, as a warm
//! transform resolves its transformer. Before that entry it looked up every
//! source fusion group in the composed-coefficient cache on every call (as
//! TensorKit's `_trace_permute!` calls the cached `fsbraid` per fusion block),
//! and a Unique trace recomposed one phase per source block (TensorKit's
//! `NoCache`). The hit must change nothing observable: values bit for bit,
//! and the structure lookups every call makes.

use std::sync::Arc;

use symtn::cache::{StructureCacheInfo, StructureCacheKind};
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{Complex64, GradedSpace, Runtime, TensorMap};

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

const CALLS: u64 = 5;
const PAIRS: &[(usize, usize)] = &[(0, 2)];

fn info(kind: StructureCacheKind) -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}

/// Lookups per kind over a window: `(hits, misses)` of the sector,
/// degeneracy and completed caches, and the composed-coefficient lookups.
#[derive(Debug, PartialEq)]
struct Lookups {
    sector: (u64, u64),
    degeneracy: (u64, u64),
    completed: (u64, u64),
    coefficient: u64,
}

fn lookups_over(calls: impl FnOnce()) -> Lookups {
    let kinds = [
        StructureCacheKind::SectorStructure,
        StructureCacheKind::DegeneracyStructure,
        StructureCacheKind::CompletedTreeTransformer,
        StructureCacheKind::TreeTransformCoefficients,
    ];
    let before = kinds.map(info);
    calls();
    let after = kinds.map(info);
    let delta = |i: usize| {
        (
            after[i].hits() - before[i].hits(),
            after[i].misses() - before[i].misses(),
        )
    };
    Lookups {
        sector: delta(0),
        degeneracy: delta(1),
        completed: delta(2),
        coefficient: delta(3).0 + delta(3).1,
    }
}

/// Runs `body` with the completed-transformer cache disabled: every compile
/// takes the uncached path. The budget is restored afterwards.
fn without_compiled_traces<T>(body: impl FnOnce() -> T) -> T {
    let kind = StructureCacheKind::CompletedTreeTransformer;
    let budget = info(kind).byte_budget();
    symtn::cache::configure_budgets([(kind, 0)]);
    let out = body();
    symtn::cache::configure_budgets([(kind, budget)]);
    out
}

fn u1_leg(leg: usize, sectors: usize) -> GradedSpace<U1FusionRule> {
    let low = -((sectors as i32 - 1) / 2);
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (0..sectors).map(|s| (U1Irrep::new(low + s as i32), 2 + (leg + s) % 3)),
    )
    .unwrap()
}

fn su2_leg(leg: usize, sectors: usize) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        (0..sectors).map(|s| (SU2Irrep::from_twice_spin(s), 2 + (leg + s) % 3)),
    )
    .unwrap()
}

fn fz2u1_leg(leg: usize, sectors: usize) -> GradedSpace<Fz2U1> {
    let low = -((sectors as i32 - 1) / 2);
    let sector = |q: i32| {
        let parity = if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        };
        ProductSector::new(parity, U1Irrep::new(q))
    };
    GradedSpace::try_new(
        Arc::new(Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap()),
        (0..sectors).map(|s| (sector(low + s as i32), 2 + (leg + s) % 3)),
    )
    .unwrap()
}

#[cfg(feature = "cgc-gen")]
fn su3_leg(leg: usize, sectors: usize) -> GradedSpace<symtn::sector::LieGroupFusionRule> {
    use std::sync::OnceLock;
    static RULE: OnceLock<Arc<symtn::sector::LieGroupFusionRule>> = OnceLock::new();
    let rule = RULE.get_or_init(|| {
        Arc::new(
            symtn::sector::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap())
                .unwrap(),
        )
    });
    let labels: [[i64; 2]; 6] = [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2]];
    GradedSpace::try_new(
        Arc::clone(rule),
        (0..sectors).map(|s| (labels[s].to_vec(), 1 + (leg + s) % 2)),
    )
    .unwrap()
}

/// `[l0, l1] <- [l0, l3]` traced over its first codomain and domain legs.
macro_rules! trace_source {
    ($runtime:expr, $leg:expr, $sectors:expr, $scalar:ty, $seed:expr) => {{
        let legs: Vec<_> = (0..4).map(|leg| $leg(leg, $sectors)).collect();
        TensorMap::<_, $scalar>::rand_with_seed(
            $runtime,
            [&legs[0], &legs[1]],
            [&legs[0], &legs[3]],
            $seed,
        )
        .unwrap()
    }};
}

/// `(source blocks, lookups of CALLS warm calls, the same with the compiled
/// traces disabled)`.
macro_rules! warm_lookups {
    ($leg:expr, $sectors:expr) => {{
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let source = trace_source!(&runtime, $leg, $sectors, f64, 2416);
        let calls = || {
            for _ in 0..CALLS {
                source.trace_pairs(PAIRS).unwrap();
            }
        };
        for _ in 0..2 {
            source.trace_pairs(PAIRS).unwrap();
        }
        let cached = lookups_over(calls);
        let uncached = without_compiled_traces(|| {
            source.trace_pairs(PAIRS).unwrap();
            lookups_over(calls)
        });
        (source.subblock_count(), cached, uncached)
    }};
}

/// What: a warm trace is one compiled-trace hit per call and no
/// composed-coefficient lookup, at any block count, for Unique (U(1)),
/// Simple (SU(2)) and checked Generic (SU(3)) fusion; its sector and
/// degeneracy lookups are those of the uncached compile: one complete
/// structure for the destination and no sector structure. The uncached
/// compile of a non-Unique rule looks its fusion groups up on every call,
/// which is what the hit removes.
#[test]
fn a_warm_trace_is_one_compiled_lookup_per_call() {
    let _serial = crate::counting_alloc::serial();
    let check = |label: &str, (blocks, cached, uncached): (usize, Lookups, Lookups), grouped| {
        assert_eq!(cached.completed, (CALLS, 0), "{label} B={blocks}");
        assert_eq!(cached.coefficient, 0, "{label} B={blocks}");
        assert_eq!(cached.sector, uncached.sector, "{label} B={blocks}");
        assert_eq!(cached.degeneracy, uncached.degeneracy, "{label} B={blocks}");
        assert_eq!(cached.sector, (0, 0), "{label} B={blocks}");
        assert_eq!(cached.degeneracy, (CALLS, 0), "{label} B={blocks}");
        if grouped {
            assert!(uncached.coefficient >= CALLS, "{label} B={blocks}");
        }
        blocks
    };
    for sectors in [3, 6] {
        check("U(1)", warm_lookups!(u1_leg, sectors), false);
        check("SU(2)", warm_lookups!(su2_leg, sectors), true);
    }
    #[cfg(feature = "cgc-gen")]
    {
        let small = check("SU(3)", warm_lookups!(su3_leg, 3), true);
        let large = check("SU(3)", warm_lookups!(su3_leg, 6), true);
        assert!(large > small);
    }
}

/// The bits of a value, for exact comparison of a reused compile.
trait Bits: Copy {
    fn bits(self) -> [u64; 2];
}

impl Bits for f64 {
    fn bits(self) -> [u64; 2] {
        [self.to_bits(), 0]
    }
}

impl Bits for Complex64 {
    fn bits(self) -> [u64; 2] {
        [self.re.to_bits(), self.im.to_bits()]
    }
}

/// `true` when `a` and `b` hold the same bits.
fn same_bits<T: Bits>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.bits() == y.bits())
}

/// Bits of the warm (hit), uncached and cold-after-clear traces of one
/// source, for `trace_pairs` and `trace_pairs_into`.
macro_rules! check_values {
    ($label:expr, $leg:expr, $sectors:expr, $scalar:ty) => {{
        let label = $label;
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let source = trace_source!(&runtime, $leg, $sectors, $scalar, 2417);
        let first = source.trace_pairs(PAIRS).unwrap();
        let hits = info(StructureCacheKind::CompletedTreeTransformer).hits();
        let warm = source.trace_pairs(PAIRS).unwrap();
        assert!(
            info(StructureCacheKind::CompletedTreeTransformer).hits() > hits,
            "{label}: the repeat must hit"
        );
        // A fresh payload each: a clone shares its storage.
        let mut into = source.trace_pairs(PAIRS).unwrap();
        source
            .trace_pairs_into(PAIRS, &mut into, <$scalar>::from(2.0), <$scalar>::from(0.5))
            .unwrap();
        let (uncached, mut uncached_into) = without_compiled_traces(|| {
            (
                source.trace_pairs(PAIRS).unwrap(),
                source.trace_pairs(PAIRS).unwrap(),
            )
        });
        without_compiled_traces(|| {
            source
                .trace_pairs_into(
                    PAIRS,
                    &mut uncached_into,
                    <$scalar>::from(2.0),
                    <$scalar>::from(0.5),
                )
                .unwrap()
        });
        symtn::cache::clear();
        let rebuilt = trace_source!(&runtime, $leg, $sectors, $scalar, 2417);
        let cold = rebuilt.trace_pairs(PAIRS).unwrap();
        let data = |t: &TensorMap<_, $scalar>| t.dense_data().unwrap().to_vec();
        assert!(!data(&warm).is_empty(), "{label}");
        assert!(same_bits(&data(&warm), &data(&first)), "{label}: warm");
        assert!(
            same_bits(&data(&warm), &data(&uncached)),
            "{label}: uncached"
        );
        assert!(same_bits(&data(&warm), &data(&cold)), "{label}: cold");
        assert!(
            same_bits(&data(&into), &data(&uncached_into)),
            "{label}: into"
        );
    }};
}

/// What: the warm trace (a compiled-trace hit) equals, bit for bit, the
/// uncached compile and a cold compile after `symtn::cache::clear`, owned
/// and into a destination, real and complex, for Unique, Simple, fermionic
/// and checked Generic rules.
#[test]
fn a_warm_trace_equals_the_uncached_and_cold_traces() {
    let _serial = crate::counting_alloc::serial();
    check_values!("U(1) f64", u1_leg, 4, f64);
    check_values!("U(1) c64", u1_leg, 4, Complex64);
    check_values!("SU(2) f64", su2_leg, 4, f64);
    check_values!("SU(2) c64", su2_leg, 4, Complex64);
    check_values!("fZ2xU(1) f64", fz2u1_leg, 4, f64);
    check_values!("fZ2xU(1) c64", fz2u1_leg, 4, Complex64);
    #[cfg(feature = "cgc-gen")]
    {
        check_values!("SU(3) f64", su3_leg, 3, f64);
        check_values!("SU(3) c64", su3_leg, 3, Complex64);
    }
}
