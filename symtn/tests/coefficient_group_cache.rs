//! The per-group composed-coefficient cache (cache 4 of #2014; #1570).
//!
//! A sector change misses the completed-transformer cache. The rebuild must
//! then recompose only the source fusion-tree groups
//! whose external sectors are new, as TensorKit's `fsbraid`/`fstranspose`
//! caches do per `FusionTreeBlock`, and the result must equal a cold
//! build's. Unique fusion never enters it (TensorKit's `NoCache`; see
//! `composed_coefficient_cache.rs`).

use std::collections::HashSet;
use std::sync::Arc;
use symtn::cache::{StructureCacheInfo, StructureCacheKind};
use symtn::sector::{SU2FusionRule, SU2Irrep, TableRule, TableSector};
use symtn::typed::Engine;
use symtn::typed::{Complex64, GradedSpace, Runtime, TensorMap};

use crate::numerics;

/// Orders this file's cache probes; the shared guard orders probes across files.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn groups() -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == StructureCacheKind::TreeTransformCoefficients)
        .unwrap()
}

fn bits(data: &[f64]) -> Vec<u64> {
    data.iter().map(|value| value.to_bits()).collect()
}

/// Source fusion-tree groups of `[leg, leg*] <- [leg, leg*]`, read from the
/// stored subblocks' external sectors. Why this is an independent count of
/// the changed groups: a group is identified by its external sectors and
/// dual flags, and its trees depend on those sectors alone, so the groups a
/// sector change adds are exactly the external-sector tuples it adds.
macro_rules! source_groups {
    ($engine:ident, $runtime:expr, $leg:expr) => {{
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let mut groups = HashSet::new();
        TensorMap::<_, f64>::from_subblock_fn(
            $engine,
            $runtime,
            [leg, &dual],
            [leg, &dual],
            |trees, _| {
                groups.insert(format!(
                    "{:?}",
                    (trees.codomain_uncoupled(), trees.domain_uncoupled())
                ));
                0.0
            },
        )
        .unwrap();
        groups
    }};
}

macro_rules! tensor {
    ($engine:ident, $runtime:expr, $leg:expr) => {{
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        TensorMap::<_, f64>::rand_with_seed($engine, $runtime, [leg, &dual], [leg, &dual], 17)
            .unwrap()
    }};
}

/// `a` warms the caches and `c` adds one sector. The transform of `c` must
/// compose exactly the groups that `c` adds and reuse every other one.
/// `$exact` asks for bit equality with a cold build.
macro_rules! check_rule {
    ($engine:ident, $label:literal, $a:expr, $c:expr, exact: $exact:expr) => {{
        let (a, c) = ($a, $c);
        let probe = Runtime::builder().build().unwrap();
        let before = source_groups!($engine, &probe, &a);
        let after = source_groups!($engine, &probe, &c);
        assert!(
            before.is_subset(&after),
            "{}: fixture must only add groups",
            $label
        );
        let changed = after.difference(&before).count();
        assert!(changed > 0 && changed < after.len(), "{}: fixture", $label);

        let operations: [(
            &str,
            &dyn Fn(&mut dyn Engine, &TensorMap<_, f64>) -> TensorMap<_, f64>,
        ); 4] = [
            ("permute", &|engine, t| {
                t.permute(engine, &[2, 0], &[3, 1]).unwrap()
            }),
            ("braid", &|engine, t| {
                t.braid(engine, &[1, 3], &[0, 2], &[0, 1, 2, 3]).unwrap()
            }),
            ("transpose", &|engine, t| {
                t.transpose(engine, &[1, 3], &[0, 2]).unwrap()
            }),
            ("adjoint permute", &|engine, t| {
                t.adjoint(engine)
                    .unwrap()
                    .permute(engine, &[2, 0], &[3, 1])
                    .unwrap()
            }),
        ];
        for (name, operation) in operations {
            let what = format!("{} {name}", $label);
            symtn::cache::clear();
            let warm = Runtime::builder().build().unwrap();
            let _ = {
                let tensor = tensor!($engine, &warm, &a);
                operation($engine, &tensor)
            };
            let groups_before = groups();
            assert_eq!(
                groups_before.misses(),
                before.len() as u64,
                "{what}: the cold transform composes every source group once"
            );

            let sector_change = {
                let tensor = tensor!($engine, &warm, &c);
                operation($engine, &tensor)
            };
            let groups = groups();
            assert_eq!(
                groups.misses() - groups_before.misses(),
                changed as u64,
                "{what}: rebuilt groups must equal changed groups"
            );
            assert_eq!(
                groups.hits() - groups_before.hits(),
                (after.len() - changed) as u64,
                "{what}: every unchanged group must hit"
            );
            assert_eq!(groups.evictions(), 0, "{what}: fixture exceeds the cache");

            symtn::cache::clear();
            let cold = Runtime::builder().build().unwrap();
            let expected = {
                let tensor = tensor!($engine, &cold, &c);
                operation($engine, &tensor)
            };
            assert_eq!(sector_change.codomain(), expected.codomain(), "{what}");
            assert_eq!(sector_change.domain(), expected.domain(), "{what}");
            if $exact {
                assert_eq!(
                    bits(
                        sector_change
                            .materialize($engine)
                            .unwrap()
                            .dense_data()
                            .unwrap()
                    ),
                    bits(expected.materialize($engine).unwrap().dense_data().unwrap()),
                    "{what}: partially reused plan differs from a cold Runtime"
                );
            } else {
                // Terms per entry: the recoupled trees of one fusion block,
                // bounded well below 64 for these rank-4 fixtures.
                numerics::assert_slices_close(
                    &what,
                    sector_change
                        .materialize($engine)
                        .unwrap()
                        .dense_data()
                        .unwrap(),
                    expected.materialize($engine).unwrap().dense_data().unwrap(),
                    64,
                );
            }
        }
    }};
}

type Su2Operation =
    dyn Fn(&mut dyn Engine, &TensorMap<SU2FusionRule, f64>) -> TensorMap<SU2FusionRule, f64>;

fn su2_leg(sectors: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        sectors
            .iter()
            .map(|&(j, d)| (SU2Irrep::from_twice_spin(j), d)),
    )
    .unwrap()
}

#[test]
fn su2_sector_change_rebuilds_only_changed_groups() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _serial = serial();
    check_rule!(engine,
        "SU(2)",
        su2_leg(&[(0, 2), (1, 2), (2, 1)]),
        su2_leg(&[(0, 2), (1, 2), (2, 1), (3, 1)]),
        exact: true
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_su3_sector_change_rebuilds_only_changed_groups() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _serial = serial();
    use symtn::sector::LieGroupFusionRule;

    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = |sectors: &[([i64; 2], usize)]| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            sectors.iter().map(|&(labels, d)| (labels.to_vec(), d)),
        )
        .unwrap()
    };
    check_rule!(engine,
        "SU(3)",
        leg(&[([1, 0], 2), ([1, 1], 1)]),
        leg(&[([0, 0], 1), ([1, 0], 2), ([1, 1], 1)]),
        // Why not bit equality: the checked-Generic SU(3) coefficients are
        // summed through fusion-symbols' randomly seeded hash maps, so two cold
        // Runtimes already differ in the last bits.
        exact: false
    );
}

/// Two rules built from equal `GroupId`s have one identity, so the second
/// rule's transform is served from the first rule's structure entries.
#[cfg(feature = "cgc-gen")]
#[test]
fn equal_lie_groups_share_structure_cache_entries() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _serial = serial();
    use symtn::sector::LieGroupFusionRule;

    let totals = || {
        symtn::cache::stats()
            .iter()
            .fold((0, 0), |(hits, misses), info| {
                (hits + info.hits(), misses + info.misses())
            })
    };
    let leg = |provider: &Arc<LieGroupFusionRule>| {
        GradedSpace::try_new(Arc::clone(provider), [(vec![1i64, 0], 2), (vec![1, 1], 1)]).unwrap()
    };
    // The quotient by the trivial subgroup is the same group from another
    // constructor.
    let (first, second) = (
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su_quotient(3, 1).unwrap())
                .unwrap(),
        ),
    );
    symtn::cache::clear();
    let runtime = Runtime::builder().build().unwrap();
    let _ = tensor!(engine, &runtime, &leg(&first))
        .permute(engine, &[2, 0], &[3, 1])
        .unwrap();
    let (hits, misses) = totals();
    assert!(misses > 0, "the first transform must build its structures");
    let _ = tensor!(engine, &runtime, &leg(&second))
        .permute(engine, &[2, 0], &[3, 1])
        .unwrap();
    let (warm_hits, warm_misses) = totals();
    assert_eq!(warm_misses, misses, "an equal group must not rebuild");
    assert!(warm_hits > hits);

    // The identity holds its group weakly: once both rules are dropped, a new
    // rule over an equal group no longer reaches their entries.
    drop((first, second));
    let third =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let _ = tensor!(engine, &runtime, &leg(&third))
        .permute(engine, &[2, 0], &[3, 1])
        .unwrap();
    let (_, cold_misses) = totals();
    assert!(
        cold_misses > warm_misses,
        "sharing ends when every rule over the group is dropped"
    );
}

#[test]
fn clear_resets_the_coefficient_cache() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _serial = serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = su2_leg(&[(0, 1), (1, 1)]);
    let _ = tensor!(engine, &runtime, &leg)
        .permute(engine, &[2, 0], &[3, 1])
        .unwrap();
    let before = groups();
    assert!(before.entries() > 0 && before.charged_bytes() > 0);
    assert!(before.charged_bytes() <= before.byte_budget());

    symtn::cache::clear();
    let after = groups();
    assert_eq!(after.entries(), 0);
    assert_eq!(after.charged_bytes(), 0);
    assert_eq!((after.hits(), after.misses()), (0, 0));
}

/// Many operations keep far more groups than the 256 structures the former
/// per-Runtime plan tier held. Every unchanged group must still hit: only
/// the byte budget binds.
#[test]
fn shared_runtime_keeps_every_unchanged_group_across_operations() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _serial = serial();
    let (a, c) = (
        su2_leg(&[(0, 2), (1, 2), (2, 1)]),
        su2_leg(&[(0, 2), (1, 2), (2, 1), (3, 1)]),
    );
    let probe = Runtime::builder().build().unwrap();
    let before = source_groups!(engine, &probe, &a).len();
    let after = source_groups!(engine, &probe, &c).len();
    let operations: [&Su2Operation; 5] = [
        &|engine, t| t.permute(engine, &[2, 0], &[3, 1]).unwrap(),
        &|engine, t| t.permute(engine, &[1, 3], &[0, 2]).unwrap(),
        &|engine, t| t.permute(engine, &[0, 2], &[1, 3]).unwrap(),
        &|engine, t| t.transpose(engine, &[1, 3], &[0, 2]).unwrap(),
        // Why not `[2, 0], [3, 1]`: on this self-adjoint space that adjoint
        // permute reads storage like the direct `[0, 2], [1, 3]` above and is
        // served by the completed-structure tier without a plan lookup.
        &|engine, t| {
            t.adjoint(engine)
                .unwrap()
                .permute(engine, &[1, 0], &[3, 2])
                .unwrap()
        },
    ];
    symtn::cache::clear();
    let runtime = Runtime::builder().build().unwrap();
    for operation in operations {
        let _ = {
            let tensor = tensor!(engine, &runtime, &a);
            operation(engine, &tensor)
        };
    }
    let warm = groups();
    for operation in operations {
        let _ = {
            let tensor = tensor!(engine, &runtime, &c);
            operation(engine, &tensor)
        };
    }
    let info = groups();
    let n = operations.len();
    assert_eq!(
        info.hits() - warm.hits(),
        (n * before) as u64,
        "every unchanged group must hit"
    );
    assert_eq!(info.misses() - warm.misses(), (n * (after - before)) as u64);
    assert_eq!(info.evictions(), 0);
    assert!(
        info.entries() > 256,
        "fixture must exceed the former plan tier's 256 entries"
    );
}

/// Operations that share a source structure but differ in the permutation,
/// the braid levels, or the orientation together with storage conjugation
/// must not share groups: each result after the others warmed the caches
/// equals a cold build bit for bit. Orientation is never varied alone: every
/// reachable adjoint-oriented caller also conjugates storage, so this covers
/// the key's orientation jointly with the conjugated binding.
#[test]
fn shared_runtime_distinguishes_operations_on_the_same_groups() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _serial = serial();
    symtn::cache::clear();
    let leg = su2_leg(&[(0, 2), (1, 2), (2, 1)]);
    let operations: [(&str, &Su2Operation); 5] = [
        ("permute p1", &|engine, t| {
            t.permute(engine, &[2, 0], &[3, 1]).unwrap()
        }),
        ("permute p2", &|engine, t| {
            t.permute(engine, &[3, 1], &[2, 0]).unwrap()
        }),
        ("transpose", &|engine, t| {
            t.transpose(engine, &[1, 3], &[0, 2]).unwrap()
        }),
        ("direct permute", &|engine, t| {
            t.permute(engine, &[1, 0], &[3, 2]).unwrap()
        }),
        ("adjoint permute", &|engine, t| {
            t.adjoint(engine)
                .unwrap()
                .permute(engine, &[1, 0], &[3, 2])
                .unwrap()
        }),
    ];
    let shared = Runtime::builder().build().unwrap();
    let warm_results = operations.map(|(_, operation)| {
        let tensor = tensor!(engine, &shared, &leg);
        operation(engine, &tensor)
    });
    for ((name, operation), warm) in operations.into_iter().zip(warm_results) {
        symtn::cache::clear();
        let cold_runtime = Runtime::builder().build().unwrap();
        let cold = {
            let tensor = tensor!(engine, &cold_runtime, &leg);
            operation(engine, &tensor)
        };
        assert_eq!(warm.codomain(), cold.codomain(), "{name}");
        assert_eq!(
            bits(warm.materialize(engine).unwrap().dense_data().unwrap()),
            bits(cold.materialize(engine).unwrap().dense_data().unwrap()),
            "{name}: shared-Runtime result differs from a cold Runtime"
        );
    }

    // Levels only matter for non-symmetric braiding.
    let fib = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 2), (TableSector::new(2), 2)],
    )
    .unwrap();
    let fib_tensor = |engine: &mut dyn Engine, runtime: &Runtime| {
        let dual = fib.try_dual().unwrap();
        TensorMap::<_, Complex64>::rand_with_seed(engine, runtime, [&fib, &dual], [&fib, &dual], 5)
            .unwrap()
    };
    let complex_bits = |engine: &mut dyn Engine, t: &TensorMap<TableRule, Complex64>| {
        t.materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect::<Vec<_>>()
    };
    symtn::cache::clear();
    let shared = Runtime::builder().build().unwrap();
    let under = fib_tensor(engine, &shared)
        .braid(engine, &[1, 3], &[0, 2], &[0, 1, 2, 3])
        .unwrap();
    let over = fib_tensor(engine, &shared)
        .braid(engine, &[1, 3], &[0, 2], &[3, 2, 1, 0])
        .unwrap();
    assert_ne!(
        complex_bits(engine, &under),
        complex_bits(engine, &over),
        "fixture: levels matter"
    );
    symtn::cache::clear();
    let cold_runtime = Runtime::builder().build().unwrap();
    let cold_over = fib_tensor(engine, &cold_runtime)
        .braid(engine, &[1, 3], &[0, 2], &[3, 2, 1, 0])
        .unwrap();
    assert_eq!(
        complex_bits(engine, &over),
        complex_bits(engine, &cold_over),
        "Fibonacci braid levels: shared-Runtime result differs from a cold Runtime"
    );
}
