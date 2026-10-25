//! `StructureSignature` is content identity (#1495).
//!
//! Every test takes one process-wide lock: the eviction, reset and
//! pointer-sharing cases observe the global structure caches. Those cases read
//! the content id from the `Debug` rendering: two signatures of the
//! same rule, placement and live runtime render differently only when their
//! content ids differ.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use symtn::cache::{StructureCacheInfo, StructureCacheKind};

fn clear_structure_caches() {
    symtn::cache::clear();
}

fn set_structure_cache_byte_budget(kind: StructureCacheKind, bytes: u64) {
    symtn::cache::configure_budgets([(kind, bytes)]);
}

fn structure_cache_info(kind: StructureCacheKind) -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::sector::{Fz2SectorLayout, PackedProductCodec, U1SectorLayout};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, StructureSignature, TensorMap};

type Fz2U1Rule = ProductFusionRule<
    FermionParityFusionRule,
    U1FusionRule,
    PackedProductCodec<Fz2SectorLayout, U1SectorLayout>,
>;

use crate::counting_alloc;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    let (value, allocs) = counting_alloc::measure(operation);
    (value, allocs.calls as usize)
}

fn hash_of(signature: &StructureSignature) -> u64 {
    let mut hasher = DefaultHasher::new();
    signature.hash(&mut hasher);
    hasher.finish()
}

fn assert_same(left: &StructureSignature, right: &StructureSignature, label: &str) {
    assert!(left == right, "{label}: {left:?} != {right:?}");
    assert_eq!(hash_of(left), hash_of(right), "{label}: hash");
}

fn assert_differ(left: &StructureSignature, right: &StructureSignature, label: &str) {
    assert!(left != right, "{label}: {left:?} == {right:?}");
}

/// One symmetry fixture: `$leg(variant)` builds an independent `GradedSpace`
/// on every call. Variant 0 is the base leg, 1 changes one degeneracy, and 2
/// changes one sector. A macro because the typed constructors' dispatch
/// bounds differ per admission mode.
macro_rules! check_fixture {
    ($label:expr, $leg:expr) => {{
        let label: &str = $label;
        let leg = $leg;
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let build = |codomain: [_; 2], domain: [_; 1]| {
            TensorMap::<_, f64>::zeros(&runtime, codomain, domain).unwrap()
        };
        let (a, b) = (leg(0), leg(0));
        let base = build([&a, &a], [&a]).structure_signature();
        assert_same(
            &base,
            &build([&b, &b], [&b]).structure_signature(),
            &format!("{label} independent"),
        );

        let dual = a.try_dual().unwrap();
        let (degeneracy, sector) = (leg(1), leg(2));
        for (other, name) in [
            (&dual, "dual"),
            (&degeneracy, "degeneracy"),
            (&sector, "sector"),
        ] {
            assert_differ(
                &base,
                &build([&a, other], [&a]).structure_signature(),
                &format!("{label} {name}"),
            );
        }

        // Same external legs, different coupled sectors and fusion-tree basis.
        let repartitioned = build([&a, &a], [&a]).repartition(1).unwrap();
        assert_differ(
            &base,
            &repartitioned.structure_signature(),
            &format!("{label} fusion-tree basis"),
        );
        assert_same(
            &base,
            &repartitioned.repartition(2).unwrap().structure_signature(),
            &format!("{label} repartition round trip"),
        );
    }};
}

#[test]
fn fixtures_are_equal_iff_structure_is_equal() {
    let _serial = crate::counting_alloc::serial();
    check_fixture!("U1", |variant| {
        let q = U1Irrep::new;
        let pairs = match variant {
            0 => [(q(-1), 2), (q(0), 1), (q(1), 3)],
            1 => [(q(-1), 2), (q(0), 2), (q(1), 3)],
            _ => [(q(-1), 2), (q(0), 1), (q(2), 3)],
        };
        GradedSpace::try_new(Arc::new(U1FusionRule), pairs).unwrap()
    });
    check_fixture!("SU2", |variant| {
        let j = SU2Irrep::from_twice_spin;
        let pairs = match variant {
            0 => [(j(0), 2), (j(1), 2), (j(2), 1)],
            1 => [(j(0), 2), (j(1), 3), (j(2), 1)],
            _ => [(j(0), 2), (j(1), 2), (j(3), 1)],
        };
        GradedSpace::try_new(Arc::new(SU2FusionRule), pairs).unwrap()
    });
    check_fixture!("fZ2xU1", |variant| {
        let rule = Arc::new(Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap());
        let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
        let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
        let pairs = match variant {
            0 => [(even(0), 2), (odd(1), 1), (odd(-1), 2)],
            1 => [(even(0), 2), (odd(1), 2), (odd(-1), 2)],
            _ => [(even(0), 2), (odd(1), 1), (even(-2), 2)],
        };
        GradedSpace::try_new(rule, pairs).unwrap()
    });
    #[cfg(feature = "cgc-gen")]
    check_fixture!("SU3 checked Generic", |variant| {
        let rule = Arc::new(
            symtn::sector::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap())
                .unwrap(),
        );
        let pairs = match variant {
            0 => [(vec![0i64, 0], 1), (vec![1, 1], 2)],
            1 => [(vec![0i64, 0], 1), (vec![1, 1], 3)],
            _ => [(vec![0i64, 0], 1), (vec![1, 0], 2)],
        };
        GradedSpace::try_new(rule, pairs).unwrap()
    });
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_rule_instance_separates_signatures() {
    let _serial = crate::counting_alloc::serial();
    // What: SU(3) and SU(4) trivial legs share hom space and block structure
    // (the unit test checks the fields), so the rule alone separates them.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let signature = |rank: usize, trivial: Vec<i64>| {
        let rule = Arc::new(
            symtn::sector::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(rank).unwrap())
                .unwrap(),
        );
        let leg = GradedSpace::try_new(rule, [(trivial, 2)]).unwrap();
        TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg])
            .unwrap()
            .structure_signature()
    };
    let su3 = signature(3, vec![0, 0]);
    assert_same(&su3, &signature(3, vec![0, 0]), "SU3 instances");
    assert_differ(&su3, &signature(4, vec![0, 0, 0]), "SU3 vs SU4");
}

fn u1_leg(charges: std::ops::RangeInclusive<i32>, degeneracy: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        charges.map(|charge| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn u1_signature(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) -> StructureSignature {
    TensorMap::<_, f64>::zeros(runtime, [leg, leg], [leg])
        .unwrap()
        .structure_signature()
}

#[test]
fn runtime_identity_separates_signatures() {
    let _serial = crate::counting_alloc::serial();
    let leg = u1_leg(-1..=1, 2);
    let first = Runtime::builder().dense_threads(1).build().unwrap();
    let second = Runtime::builder().dense_threads(1).build().unwrap();
    let signature = u1_signature(&first, &leg);
    assert_same(
        &signature,
        &u1_signature(&first.clone(), &leg),
        "shared runtime",
    );
    assert_differ(&signature, &u1_signature(&second, &leg), "second runtime");

    // A dead Runtime's signature never matches a Runtime built afterwards.
    drop(first);
    let third = Runtime::builder().dense_threads(1).build().unwrap();
    assert_differ(&signature, &u1_signature(&third, &leg), "after drop");
}

#[test]
fn equal_across_clear_structure_caches() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let before = u1_signature(&runtime, &u1_leg(-2..=2, 3));
    clear_structure_caches();
    let after = u1_signature(&runtime, &u1_leg(-2..=2, 3));
    assert_ne!(
        format!("{before:?}"),
        format!("{after:?}"),
        "premise: fresh content"
    );
    assert_same(&before, &after, "reset");
}

#[test]
fn equal_across_complete_owner_eviction() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let before = u1_signature(&runtime, &u1_leg(-3..=3, 5));
    let cache = StructureCacheKind::DegeneracyStructure;
    let before_info = structure_cache_info(cache);
    let budget = before_info.byte_budget();
    // Evict the actual complete owner while the signature keeps its content
    // alive. Rebuilding must preserve semantic equality, not the process id.
    set_structure_cache_byte_budget(cache, 0);
    assert_eq!(structure_cache_info(cache).entries(), 0);
    assert!(structure_cache_info(cache).evictions() > before_info.evictions());
    set_structure_cache_byte_budget(cache, budget);
    let after = u1_signature(&runtime, &u1_leg(-3..=3, 5));
    assert_ne!(
        format!("{before:?}"),
        format!("{after:?}"),
        "premise: evicted"
    );
    assert_same(&before, &after, "eviction");
}

#[test]
fn equal_for_oversized_structures_above_the_complete_cache_budget() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    // An explicit small budget rejects this otherwise ordinary complete layout.
    // Rejected layouts receive distinct process ids but remain semantically
    // equal while each signature owns its independently constructed content.
    let leg = u1_leg(-3..=3, 1);
    let cache = StructureCacheKind::DegeneracyStructure;
    let budget = structure_cache_info(cache).byte_budget();
    set_structure_cache_byte_budget(cache, 1);
    let signature = || {
        TensorMap::<_, f64>::zeros(&runtime, [&leg], [&leg])
            .unwrap()
            .structure_signature()
    };
    let rejected = structure_cache_info(StructureCacheKind::DegeneracyStructure).rejections();
    let before = signature();
    let after = signature();
    set_structure_cache_byte_budget(cache, budget);
    assert!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure).rejections() >= rejected + 2
    );
    assert_ne!(
        format!("{before:?}"),
        format!("{after:?}"),
        "premise: independently rejected"
    );
    assert_same(&before, &after, "oversized");
}

#[test]
fn shared_structure_comparison_does_not_allocate() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = u1_leg(-2..=2, 2);
    let first = u1_signature(&runtime, &leg);
    let second = u1_signature(&runtime, &leg);
    assert_eq!(
        format!("{first:?}"),
        format!("{second:?}"),
        "premise: shared"
    );
    let (equal, allocations) = measured(|| first == second);
    assert!(equal);
    assert_eq!(allocations, 0, "pointer fast path");
    let (_, allocations) = measured(|| hash_of(&first));
    assert_eq!(allocations, 0, "hash");
}
