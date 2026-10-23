use std::collections::BTreeSet;
use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
#[cfg(feature = "cgc-gen")]
use symtn::sector::{CheckedGenericRigidSymbols, TypedSectorAdmission};
use symtn::sector::{
    Fz2SectorLayout, PackedProductCodec, ProductSectorLayout, Su2SectorLayout, U1SectorLayout,
};
use symtn::typed::TensorScalar;
use symtn::typed::{Complex32, Complex64, GradedSpace, Runtime, SectorSpectrum, TensorMap};

type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Su2Codec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
type Fz2U1Su2Rule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, Fz2U1Su2Codec>;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize) {
    let (value, allocs) = counting_alloc::measure(operation);
    (value, allocs.calls as usize)
}

fn non_abelian_space() -> GradedSpace<Fz2U1Su2Rule> {
    let rule = Arc::new(
        Fz2U1Su2Rule::try_new(
            Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    );
    let label = |parity: u8, charge: i32, twice_spin: usize| {
        product_sector(
            product_sector(
                if parity == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                },
                U1Irrep::new(charge),
            ),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    GradedSpace::try_new(
        rule,
        [
            (label(0, -2, 0), 4),
            (label(0, 1, 2), 3),
            (label(1, -1, 1), 4),
            (label(1, 2, 3), 2),
        ],
    )
    .unwrap()
}

#[test]
fn warmed_non_abelian_inner_and_norm_do_not_allocate() {
    // What: cold region compilation is observable, while explicit warm-up
    // leaves both public reductions allocation-free on the caller thread.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let lhs: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 282_401).unwrap();
    let rhs: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 282_402).unwrap();

    let (cold, cold_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    eprintln!("cold coupled-region initialization: {cold_allocations} allocations");
    black_box(cold);
    black_box(lhs.norm(2.0).unwrap());

    let (inner, inner_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    let (norm, norm_allocations) = measured(|| lhs.norm(2.0).unwrap());
    black_box((inner, norm));

    assert_eq!(inner_allocations, 0);
    assert_eq!(norm_allocations, 0);

    // Out-of-range payloads take the rescaling passes, which add none either.
    for scale in [1e200, 1e-200] {
        let extreme = lhs.scale(Complex64::new(scale, 0.0)).unwrap();
        black_box(extreme.norm(2.0).unwrap());
        let (value, allocations) = measured(|| extreme.norm(2.0).unwrap());
        assert!(value.is_finite() && value > 0.0, "{scale:e}: {value:e}");
        assert_eq!(allocations, 0, "rescaled norm at {scale:e}");
        let (value, allocations) = measured(|| extreme.norm(3.0).unwrap());
        assert!(value.is_finite() && value > 0.0, "{scale:e}: {value:e}");
        assert_eq!(allocations, 0, "rescaled norm(3) at {scale:e}");
    }
}

#[test]
fn warmed_non_abelian_trace_does_not_allocate() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let owned: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space, &space], 282_403).unwrap();
    let lazy = owned.adjoint().unwrap();

    black_box(owned.tr().unwrap());
    black_box(lazy.tr().unwrap());
    for (row, (value, allocations)) in [
        ("owned trace", measured(|| owned.tr().unwrap())),
        ("lazy trace", measured(|| lazy.tr().unwrap())),
    ] {
        black_box(value);
        assert_eq!(allocations, 0, "{row}");
    }
}

#[test]
fn warmed_lazy_adjoint_inner_does_not_allocate_in_mixed_or_double_orientation() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = non_abelian_space();
    let lhs_parent: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_301).unwrap();
    let rhs_parent: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_302).unwrap();
    let owned: TensorMap<Fz2U1Su2Rule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space, &space], 666_303).unwrap();
    let warm_lhs = lhs_parent.adjoint().unwrap();
    let warm_rhs = rhs_parent.adjoint().unwrap();
    black_box(warm_lhs.inner(&owned).unwrap());
    black_box(owned.inner(&warm_lhs).unwrap());
    black_box(warm_lhs.inner(&warm_rhs).unwrap());

    let lhs_mixed_left = lhs_parent.adjoint().unwrap();
    let lhs_mixed_right = lhs_parent.adjoint().unwrap();
    let lhs_double = lhs_parent.adjoint().unwrap();
    let rhs_double = rhs_parent.adjoint().unwrap();
    for (value, allocations) in [
        measured(|| lhs_mixed_left.inner(&owned).unwrap()),
        measured(|| owned.inner(&lhs_mixed_right).unwrap()),
        measured(|| lhs_double.inner(&rhs_double).unwrap()),
    ] {
        black_box(value);
        // Zero allocations also rules out an operation-local materialization.
        assert_eq!(allocations, 0);
    }
}

#[test]
fn warmed_compact_dense_inner_does_not_allocate() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let space =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 3), (spin_half, 2)]).unwrap();
    let compact = TensorMap::diagonal(
        &runtime,
        &space,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(1.0, 0.5); 3],
            },
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(2.0, -0.5); 2],
            },
        ],
    )
    .unwrap();
    let dense: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space], 1_616).unwrap();
    let lazy = dense.adjoint().unwrap();

    for (value, allocations) in [
        measured(|| compact.inner(&dense).unwrap()),
        measured(|| dense.inner(&compact).unwrap()),
        measured(|| compact.inner(&lazy).unwrap()),
        measured(|| lazy.inner(&compact).unwrap()),
    ] {
        black_box(value);
        assert_eq!(allocations, 0);
    }
}

// The shared checked Generic toy: outer multiplicity two on `X (x) X -> X`
// and an allocation-free `dim`, so the rows below measure the reduction
// owners and not the provider.
#[path = "../../tests/support/toy_rules.rs"]
mod toy_rules;

use toy_rules::{GenericLabel, GenericToy};

fn block_value<D: TensorScalar>(
    trees: &symtn::typed::BlockFusionTrees<GenericLabel>,
    indices: &[usize],
) -> D {
    let vertex = trees.codomain_vertices()[0].get() as f64;
    D::from_real(
        vertex
            + indices.iter().enumerate().fold(0.0, |acc, (axis, &index)| {
                acc + (index as f64 + 1.0) * (axis as f64 + 1.0)
            }),
    )
}

/// Coupled sectors `G` of a tensor; each one is exactly one packed region.
macro_rules! coupled_sector_count {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        (0..tensor.subblock_count())
            .map(|index| {
                tensor
                    .subblock_fusion_trees(index)
                    .unwrap()
                    .coupled()
                    .to_owned()
            })
            .collect::<BTreeSet<_>>()
            .len()
    }};
}

#[test]
fn warmed_checked_generic_reductions_do_not_allocate() {
    // What: checked Generic reductions take their `dim(c)` weights
    // through one fallible closure and never build a per-call weight map
    // (a std `HashMap`, #1219), for owned and lazy-adjoint inputs alike.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(GenericToy);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(GenericLabel::Vacuum, 1), (GenericLabel::X, 2)],
    )
    .unwrap();
    let lhs: TensorMap<GenericToy, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], block_value).unwrap();
    let rhs: TensorMap<GenericToy, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            block_value::<Complex64>(trees, indices) * Complex64::new(0.5, -1.5)
        })
        .unwrap();
    assert!(lhs.subblock_count() > coupled_sector_count!(lhs));
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();

    let (cold, cold_allocations) = measured(|| lhs.inner(&rhs).unwrap());
    eprintln!("cold checked coupled-region initialization: {cold_allocations} allocations");
    black_box(cold);
    black_box(lhs.norm(2.0).unwrap());
    black_box(lazy_lhs.inner(&lazy_rhs).unwrap());
    black_box(lazy_lhs.inner(&rhs).unwrap());
    black_box(lazy_lhs.norm(2.0).unwrap());
    black_box(lhs.tr().unwrap());
    black_box(lazy_lhs.tr().unwrap());

    for (row, (value, allocations)) in [
        ("owned inner", measured(|| lhs.inner(&rhs).unwrap())),
        (
            "owned norm",
            measured(|| Complex64::from(lhs.norm(2.0).unwrap())),
        ),
        (
            "lazy-lazy inner",
            measured(|| lazy_lhs.inner(&lazy_rhs).unwrap()),
        ),
        (
            "lazy-owned inner",
            measured(|| lazy_lhs.inner(&rhs).unwrap()),
        ),
        (
            "owned-lazy inner",
            measured(|| rhs.inner(&lazy_lhs).unwrap()),
        ),
        (
            "lazy norm",
            measured(|| Complex64::from(lazy_lhs.norm(2.0).unwrap())),
        ),
        ("owned trace", measured(|| lhs.tr().unwrap())),
        ("lazy trace", measured(|| lazy_lhs.tr().unwrap())),
    ] {
        black_box(value);
        assert_eq!(allocations, 0, "{row}");
    }

    // The rescaled checked norm neither allocates nor loses the value: scaling
    // commutes with the norm, so `norm(s * t) = s * norm(t)` within the
    // workspace rule taken relative to the result (the payload's entry count
    // bounds the terms of the sum).
    let norm = lhs.norm(2.0).unwrap();
    for scale in [1e200, 1e-200] {
        let extreme = lhs.scale(Complex64::new(scale, 0.0)).unwrap();
        let lazy = extreme.adjoint().unwrap();
        black_box((extreme.norm(2.0).unwrap(), lazy.norm(2.0).unwrap()));
        for (row, tensor) in [("owned", &extreme), ("lazy", &lazy)] {
            let (value, allocations) = measured(|| tensor.norm(2.0).unwrap());
            assert_eq!(allocations, 0, "{row} rescaled norm at {scale:e}");
            let want = scale * norm;
            let bound =
                32.0 * (lhs.dense_data().unwrap().len() as f64).sqrt() * f64::EPSILON * want;
            assert!(
                (value - want).abs() <= bound,
                "{row} {scale:e}: {value:e} against {want:e}"
            );
        }
    }
}

/// SU(N) checked `dim` allocates per query (`symtn-sectors/src/sun.rs::
/// decode_dynkin`'s `Vec<i64>` plus fusion-symbols' own irrep decode), so the
/// reductions still pay the provider's cost once per coupled sector; the
/// reduction owners themselves add none on top of the G provider queries
/// (#1219).
#[cfg(feature = "cgc-gen")]
#[test]
fn warmed_su3_checked_inner_and_norm_allocate_only_through_the_provider() {
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)]).unwrap();
    let lhs: TensorMap<SUNFusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        })
        .unwrap();
    let rhs: TensorMap<SUNFusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            2.0 * trees.domain_vertices()[0].get() as f64 - indices[0] as f64
        })
        .unwrap();
    let coupled = (0..lhs.subblock_count())
        .map(|index| {
            lhs.subblock_fusion_trees(index)
                .unwrap()
                .coupled()
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let sectors = coupled.len();
    assert!(sectors >= 2);
    // The provider's own per-query cost, summed over the G coupled sectors.
    let provider_allocations: usize = coupled
        .iter()
        .map(|label| {
            let sector = provider.try_encode_label(label).unwrap();
            black_box(provider.try_sqrt_dim_scalar(sector).unwrap());
            let (weight, allocations) = measured(|| provider.try_sqrt_dim_scalar(sector).unwrap());
            black_box(weight);
            allocations
        })
        .sum();
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();
    black_box(lhs.inner(&rhs).unwrap());
    black_box(lhs.norm(2.0).unwrap());
    black_box(lazy_lhs.inner(&lazy_rhs).unwrap());
    black_box(lazy_lhs.norm(2.0).unwrap());

    for (row, (value, allocations)) in [
        ("owned inner", measured(|| lhs.inner(&rhs).unwrap())),
        ("owned norm", measured(|| lhs.norm(2.0).unwrap())),
        (
            "lazy inner",
            measured(|| lazy_lhs.inner(&lazy_rhs).unwrap()),
        ),
        ("lazy norm", measured(|| lazy_lhs.norm(2.0).unwrap())),
    ] {
        black_box(value);
        eprintln!(
            "su3 {row}: {allocations} allocations for {sectors} coupled sectors \
             ({provider_allocations} from the provider's dim queries)"
        );
        assert_eq!(allocations, provider_allocations, "{row}");
    }
}

/// Checked-Generic `norm` of single-precision payloads accumulates in `f64`
/// (#1315, #1344) and rescales from that wide sum (#1437).
///
/// Oracle: the same `f32` entries widened exactly into an `f64` tensor, whose
/// norm cannot overflow or underflow at these magnitudes. A single-precision
/// payload therefore must agree to `f64` rounding — the workspace rule at
/// `f64` epsilon, relative to the result, with the entry count as `terms` —
/// which a sum narrowed to `f32` (relative error near `2^-24`, or far worse
/// once `|t|²` is `f32`-subnormal) cannot meet.
#[test]
fn checked_generic_single_precision_norm_accumulates_wide() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(GenericToy);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(GenericLabel::Vacuum, 1), (GenericLabel::X, 2)],
    )
    .unwrap();

    let check = |what: &str,
                 value: &dyn Fn(
        &symtn::typed::BlockFusionTrees<GenericLabel>,
        &[usize],
    ) -> Complex32| {
        let real32: TensorMap<GenericToy, f32> =
            TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
                value(trees, indices).re
            })
            .unwrap();
        let real64: TensorMap<GenericToy, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
                f64::from(value(trees, indices).re)
            })
            .unwrap();
        let complex32: TensorMap<GenericToy, Complex32> =
            TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
                value(trees, indices)
            })
            .unwrap();
        let complex64: TensorMap<GenericToy, Complex64> =
            TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
                let z = value(trees, indices);
                Complex64::new(z.re.into(), z.im.into())
            })
            .unwrap();
        let terms = real64.dense_data().unwrap().len() as f64;
        for (row, got, want) in [
            ("f32", real32.norm(2.0).unwrap(), real64.norm(2.0).unwrap()),
            (
                "c32",
                complex32.norm(2.0).unwrap(),
                complex64.norm(2.0).unwrap(),
            ),
        ] {
            let bound = 32.0 * terms.sqrt() * f64::EPSILON * want;
            assert!(
                want > 0.0 && (got - want).abs() <= bound,
                "{what} {row}: {got:e} against {want:e} (tolerance {bound:e})"
            );
        }
    };

    for scale in [1.0_f32, 1e-24, 1e-25] {
        check(&format!("block values x{scale:e}"), &|trees, indices| {
            let x = block_value::<f32>(trees, indices) * scale;
            Complex32::new(x, 0.5 * x)
        });
    }
    // One `8192` among ones: `8192^2 = 2^26` has `f32` spacing 8, so an `f32`
    // rounding of the sum drops most of the unit terms.
    check("8192 among ones", &|_, indices| {
        let x = if indices.iter().all(|&index| index == 0) {
            8192.0
        } else {
            1.0
        };
        Complex32::new(x, 1.0)
    });
}

/// Checked Generic compact reductions read the stored spectrum: warmed, they
/// allocate nothing, so no operation-local dense payload exists (#1867). The
/// compact plus lazy `axpby` allocates its one dense result and never a
/// densified copy of the compact operand.
#[test]
fn warmed_checked_generic_compact_reductions_do_not_densify() {
    const K: usize = 16;
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(GenericToy), [(GenericLabel::X, K)]).unwrap();
    let compact = TensorMap::<_, Complex64>::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: GenericLabel::X,
            values: (0..K)
                .map(|i| Complex64::new(1.0 + i as f64, -0.5 * i as f64))
                .collect(),
        }],
    )
    .unwrap();
    let dense: TensorMap<GenericToy, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg], 1_867).unwrap();
    let lazy = dense.adjoint().unwrap();
    // One `K x K` complex payload: the result. A densified compact operand
    // would be a second allocation of exactly this size.
    let payload = K * K * std::mem::size_of::<Complex64>();
    for (row, call) in [
        (
            "axpby(compact, lazy)",
            &(|| compact.axpby(Complex64::new(0.5, 0.0), &lazy, Complex64::new(2.0, 0.0)))
                as &dyn Fn() -> _,
        ),
        ("axpby(lazy, compact)", &|| {
            lazy.axpby(Complex64::new(2.0, 0.0), &compact, Complex64::new(0.5, 0.0))
        }),
    ] {
        black_box(call().unwrap());
        let (value, allocations) = counting_alloc::measure_matching(payload..=payload, call);
        black_box(value.unwrap());
        assert_eq!(allocations.matched_calls, 1, "{row}");
    }

    let rows: [(&str, &dyn Fn() -> Complex64); 10] = [
        ("compact inner", &|| compact.inner(&compact).unwrap()),
        ("compact-dense inner", &|| compact.inner(&dense).unwrap()),
        ("dense-compact inner", &|| dense.inner(&compact).unwrap()),
        ("compact-lazy inner", &|| compact.inner(&lazy).unwrap()),
        ("lazy-compact inner", &|| lazy.inner(&compact).unwrap()),
        ("norm(1)", &|| compact.norm(1.0).unwrap().into()),
        ("norm(2)", &|| compact.norm(2.0).unwrap().into()),
        ("norm(3)", &|| compact.norm(3.0).unwrap().into()),
        ("norm(Inf)", &|| compact.norm(f64::INFINITY).unwrap().into()),
        ("tr", &|| compact.tr().unwrap()),
    ];
    for (row, call) in rows {
        black_box(call());
        let (value, allocations) = measured(call);
        black_box(value);
        assert_eq!(allocations, 0, "{row}");
    }
}

#[test]
fn warmed_complex_symbol_reductions_do_not_allocate() {
    // What: the complex-symbol lane (#2281) reads the real quantum dimension
    // through the same seam, so warm Ising `inner`, `norm` and `tr` stay
    // allocation-free on the caller thread, owned and lazy.
    use symtn::sector::{TableRule, TableSector};
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = GradedSpace::try_new(
        Arc::new(TableRule::ising()),
        [
            (TableSector::new(1), 3),
            (TableSector::new(2), 2),
            (TableSector::new(3), 4),
        ],
    )
    .unwrap();
    let lhs: TensorMap<TableRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space, &space], 228_101).unwrap();
    let rhs: TensorMap<TableRule, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space, &space], 228_102).unwrap();
    let lazy = lhs.adjoint().unwrap();

    black_box(lhs.inner(&rhs).unwrap());
    black_box(lhs.norm(2.0).unwrap());
    black_box(lhs.tr().unwrap());
    black_box(lazy.tr().unwrap());
    black_box(lazy.norm(2.0).unwrap());
    for (row, allocations) in [
        ("inner", measured(|| lhs.inner(&rhs).unwrap()).1),
        ("norm", measured(|| lhs.norm(2.0).unwrap()).1),
        ("tr", measured(|| lhs.tr().unwrap()).1),
        ("lazy tr", measured(|| lazy.tr().unwrap()).1),
        ("lazy norm", measured(|| lazy.norm(2.0).unwrap()).1),
    ] {
        assert_eq!(allocations, 0, "{row}");
    }
}
