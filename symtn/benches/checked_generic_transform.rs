//! Public checked-Generic transform timing (SU(3)/SU(4) permute, braid,
//! repartition, plus SU(2) controls: `SU2FusionRule` against
//! `LieGroupFusionRule(su(2))` in checked Generic (`su2cg`) and
//! multiplicity-free (`su2mf`) mode for permute, braid, repartition and
//! contract). Run with `cargo bench -p symtn
//! --features cgc-gen --bench checked_generic_transform`.
//!
//! Each case runs in a child process so provider and Runtime state start
//! cold. Allocation counts cover requested bytes on the caller thread.
//! Timings are evidence, never a CI threshold.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use std::sync::Arc;

use symtn::sector::{
    CheckedGenericAdmissionMode, LieGroupFusionRule, MultiplicityFreeAdmissionMode, SU2FusionRule,
    SU2Irrep,
};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn measure_value<T>(f: impl FnOnce() -> T) -> (T, usize, usize, u128) {
    let ((output, elapsed), allocs) = counting_alloc::measure(|| {
        let started = std::time::Instant::now();
        let output = f();
        (output, started.elapsed().as_nanos())
    });
    (
        output,
        allocs.calls as usize,
        allocs.bytes as usize,
        elapsed,
    )
}

fn assert_same_checked_tensor(
    actual: &TensorMap<LieGroupFusionRule, f64>,
    expected: &TensorMap<LieGroupFusionRule, f64>,
) {
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    assert_eq!(
        actual.dense_data().unwrap().len(),
        expected.dense_data().unwrap().len()
    );
    for index in 0..actual.subblock_count() {
        assert_eq!(
            actual.subblock_fusion_trees(index).unwrap(),
            expected.subblock_fusion_trees(index).unwrap()
        );
    }
    for (actual, expected) in actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
    {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}

/// The SU(2) control cases: `SU2FusionRule` (MF) against
/// `LieGroupFusionRule(su(2))` (checked Generic) on the same spaces.
/// `small` is the original control (`[1/2, 1/2 | 0 + 1]`); `wide` has
/// spins `0..=2`, two copies each, on all four legs.
fn su2_control_cases() -> impl Iterator<Item = String> {
    ["su2", "su2cg", "su2mf"].into_iter().flat_map(|rule| {
        ["small", "wide"].into_iter().flat_map(move |fixture| {
            ["permute", "braid", "repartition", "contract"]
                .into_iter()
                .map(move |operation| format!("{rule}_permute_control/{fixture}/{operation}"))
        })
    })
}

// A macro rather than a generic function: the two rules dispatch through
// different mode traits, and the bench only needs the same call sequence.
macro_rules! su2_control {
    ($engine:expr, $case:expr, $fixture:expr, $operation:expr, $rule:ty, $make:expr, $label:expr) => {{
        let engine = $engine;
        let case = $case;
        let label = $label;
        let (provider, provider_allocations, provider_bytes, provider_ns) =
            measure_value(|| Arc::new($make));
        let runtime = Runtime::builder().build().unwrap();
        let space = |sectors: &[(usize, usize)]| {
            GradedSpace::<$rule>::try_new(
                Arc::clone(&provider),
                sectors.iter().map(|&(twice, degeneracy)| (label(twice), degeneracy)),
            )
            .unwrap()
        };
        let (contract_codomain, contract_domain) = (&[1usize, 0][..], &[2usize, 3][..]);
        let (source, other, contract_spec_axes, transform_axes) = {
            let ((source, other), source_allocations, source_bytes, source_ns) =
                measure_value(|| match $fixture {
                    "small" => {
                        let half = space(&[(1, 1)]);
                        let coupled = space(&[(0, 1), (2, 1)]);
                        let source: TensorMap<$rule, f64> = TensorMap::rand_with_seed(
                            engine,
                            &runtime,
                            [&half, &half],
                            [&coupled],
                            783,
                        )
                        .unwrap();
                        let other: TensorMap<$rule, f64> = TensorMap::rand_with_seed(
                            engine,
                            &runtime,
                            [&coupled],
                            [&half, &half],
                            784,
                        )
                        .unwrap();
                        (source, other)
                    }
                    "wide" => {
                        let wide = space(&[(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]);
                        let source: TensorMap<$rule, f64> = TensorMap::rand_with_seed(
                            engine,
                            &runtime,
                            [&wide, &wide],
                            [&wide, &wide],
                            783,
                        )
                        .unwrap();
                        let other: TensorMap<$rule, f64> = TensorMap::rand_with_seed(
                            engine,
                            &runtime,
                            [&wide, &wide],
                            [&wide, &wide],
                            784,
                        )
                        .unwrap();
                        (source, other)
                    }
                    _ => panic!("unknown fixture: {}", $fixture),
                });
            println!(
                "case={case} phase=construct provider_ns={provider_ns} provider_allocations={provider_allocations} provider_bytes={provider_bytes} source_ns={source_ns} source_allocations={source_allocations} source_bytes={source_bytes} subblocks={}",
                source.subblock_count()
            );
            match $fixture {
                // A[a0, a1 | a2] B[b0 | b1, b2], a2 = b0 -> C[a1, a0 | b1, b2].
                "small" => (source, other, (&[2usize][..], &[0usize][..]), (&[1usize, 0][..], &[2usize][..], &[0usize, 1, 2][..], 1usize)),
                // A[a0, a1 | a2, a3] B[b0, b1 | b2, b3], (a2, a3) = (b1, b0).
                _ => (source, other, (&[2usize, 3][..], &[1usize, 0][..]), (&[1usize, 0][..], &[3usize, 2][..], &[0usize, 1, 2, 3][..], 1usize)),
            }
        };
        let (codomain_axes, domain_axes, levels, split) = transform_axes;
        let spec = symtn::typed::ContractSpec {
            lhs: contract_spec_axes.0,
            rhs: contract_spec_axes.1,
            codomain: contract_codomain,
            domain: contract_domain,
        };
        let mut apply = || match $operation {
            "permute" => source.permute(engine, codomain_axes, domain_axes).unwrap(),
            "braid" => source.braid(engine, codomain_axes, domain_axes, levels).unwrap(),
            "repartition" => source.repartition(engine, split).unwrap(),
            "contract" => source.contract(engine, &other, &spec).unwrap(),
            _ => panic!("unknown operation: {}", $operation),
        };
        let (first, first_allocations, first_bytes, first_ns) = measure_value(&mut apply);
        let checksum: f64 = first.dense_data().unwrap().iter().map(|x| x * x).sum();
        let mut repeat_ns = Vec::with_capacity(15);
        let mut repeat_allocations = Vec::with_capacity(15);
        let mut repeat_bytes = Vec::with_capacity(15);
        for _ in 0..15 {
            let (repeated, allocations, bytes, ns) = measure_value(&mut apply);
            assert_eq!(repeated.dense_data().unwrap(), first.dense_data().unwrap());
            repeat_ns.push(ns);
            repeat_allocations.push(allocations);
            repeat_bytes.push(bytes);
        }
        let mut sorted_ns = repeat_ns.clone();
        sorted_ns.sort_unstable();
        println!(
            "case={case} phase=public_first ns={first_ns} allocations={first_allocations} requested_bytes={first_bytes} output_subblocks={} checksum={checksum:.12e}",
            first.subblock_count()
        );
        println!(
            "case={case} phase=public_repeat samples_ns={repeat_ns:?} median_ns={} allocations={repeat_allocations:?} requested_bytes={repeat_bytes:?}",
            sorted_ns[sorted_ns.len() / 2]
        );
    }};
}

fn main() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    const CASE_ENV: &str = "SYMTN_CHECKED_GENERIC_MEASUREMENT_CASE";

    let Some(case) = std::env::var_os(CASE_ENV) else {
        for case in [
            "su3_permute",
            "su3_braid",
            "su3_repartition",
            "su4_permute",
            "su4_braid",
            "su4_repartition",
        ]
        .into_iter()
        .map(String::from)
        .chain(su2_control_cases())
        {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .env(CASE_ENV, &case)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "case={case} status={}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            print!("{stdout}");
        }
        return;
    };
    let case = case.to_str().unwrap();
    println!("case={case} allocation_scope=caller_thread_requested");

    if let Some(control) = case
        .strip_prefix("su2_permute_control/")
        .map(|rest| ("su2", rest))
        .or_else(|| {
            case.strip_prefix("su2cg_permute_control/")
                .map(|rest| ("su2cg", rest))
        })
        .or_else(|| {
            case.strip_prefix("su2mf_permute_control/")
                .map(|rest| ("su2mf", rest))
        })
    {
        let (rule, rest) = control;
        let (fixture, operation) = rest.split_once('/').unwrap();
        match rule {
            "su2" => su2_control!(
                engine,
                case,
                fixture,
                operation,
                SU2FusionRule,
                SU2FusionRule,
                |twice: usize| SU2Irrep::from_twice_spin(twice)
            ),
            "su2mf" => {
                // Labels come from a category built outside the timed regions.
                let su2 = fusion_symbols::lie::LieGroup::new(
                    fusion_symbols::lie::GroupId::su(2).unwrap(),
                )
                .unwrap();
                su2_control!(
                    engine,
                    case,
                    fixture,
                    operation,
                    LieGroupFusionRule<MultiplicityFreeAdmissionMode>,
                    LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(
                        fusion_symbols::lie::GroupId::su(2).unwrap()
                    )
                    .unwrap(),
                    |twice: usize| su2.irrep(&[twice as i64]).unwrap()
                )
            }
            _ => {
                let su2 = fusion_symbols::lie::LieGroup::new(
                    fusion_symbols::lie::GroupId::su(2).unwrap(),
                )
                .unwrap();
                su2_control!(
                    engine,
                    case,
                    fixture,
                    operation,
                    LieGroupFusionRule,
                    LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
                        fusion_symbols::lie::GroupId::su(2).unwrap()
                    )
                    .unwrap(),
                    |twice: usize| su2.irrep(&[twice as i64]).unwrap()
                )
            }
        }
        return;
    }

    let (n, adjoint, operation) = match case {
        "su3_permute" => (3, su_irrep(&[1, 1]), "permute"),
        "su3_braid" => (3, su_irrep(&[1, 1]), "braid"),
        "su3_repartition" => (3, su_irrep(&[1, 1]), "repartition"),
        "su4_permute" => (4, su_irrep(&[1, 0, 1]), "permute"),
        "su4_braid" => (4, su_irrep(&[1, 0, 1]), "braid"),
        "su4_repartition" => (4, su_irrep(&[1, 0, 1]), "repartition"),
        _ => panic!("unknown measurement case: {case}"),
    };
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(
        LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
            fusion_symbols::lie::GroupId::su(n).unwrap(),
        )
        .unwrap(),
    );
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let mut apply = |source: &TensorMap<LieGroupFusionRule, f64>| match operation {
        "permute" => source.permute(engine, &[1, 0], &[2]).unwrap(),
        "braid" => source.braid(engine, &[1, 0], &[2], &[0, 1, 2]).unwrap(),
        "repartition" => source.repartition(engine, 1).unwrap(),
        _ => unreachable!(),
    };

    let runtime_before = completed_transformers();
    let (first, first_allocations, first_bytes, first_ns) = measure_value(|| apply(&source));
    let runtime_after_first = completed_transformers();
    let mut repeat_ns = Vec::with_capacity(7);
    let mut repeat_allocations = Vec::with_capacity(7);
    let mut repeat_bytes = Vec::with_capacity(7);
    for _ in 0..7 {
        let (repeated, allocations, bytes, ns) = measure_value(|| apply(&source));
        assert_same_checked_tensor(&repeated, &first);
        assert!(std::ptr::eq(repeated.provider(), provider.as_ref()));
        repeat_ns.push(ns);
        repeat_allocations.push(allocations);
        repeat_bytes.push(bytes);
    }
    let runtime_after_repeat = completed_transformers();
    let mut sorted_ns = repeat_ns.clone();
    sorted_ns.sort_unstable();
    println!(
        "case={case} phase=public_first_after_source_construction coefficient_caches=cold ns={first_ns} allocations={first_allocations} requested_bytes={first_bytes} runtime_before={runtime_before:?} runtime_after={runtime_after_first:?}"
    );
    println!(
        "case={case} phase=public_repeat_provider_warm samples_ns={repeat_ns:?} median_ns={} allocations={repeat_allocations:?} requested_bytes={repeat_bytes:?} runtime_after={runtime_after_repeat:?}",
        sorted_ns[sorted_ns.len() / 2]
    );
    assert!(runtime_after_first.entries() > runtime_before.entries());
    assert!(runtime_after_repeat.hits() > runtime_after_first.hits());
    assert!(std::ptr::eq(first.provider(), provider.as_ref()));
}

fn su_irrep(labels: &[i64]) -> symtn::sector::LieIrrep {
    fusion_symbols::lie::LieGroup::new(fusion_symbols::lie::GroupId::su(labels.len() + 1).unwrap())
        .unwrap()
        .irrep(labels)
        .unwrap()
}
