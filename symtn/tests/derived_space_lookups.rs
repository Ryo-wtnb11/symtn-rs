//! A warm derived output space costs one structure-cache lookup: the complete
//! (degeneracy) structure, as TensorKit's `degeneracystructure`
//! (`spaces/structure.jl:114`), whose `sectorstructure` runs only on a miss.
//! The sector structure is never consulted on a warm call. Every test in this
//! target takes the shared serial guard.

use std::sync::Arc;
use symtn::typed::Engine;

use symtn::cache::{StructureCacheInfo, StructureCacheKind};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Eigh, GradedSpace, HermitianTol, Qr, Svd, TensorMap};

use crate::fixtures::host_runtime;

fn info(kind: StructureCacheKind) -> StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}

/// Bit pattern of the materialized dense data of each factor, read outside
/// the measured call.
macro_rules! bits {
    ($engine:ident, $($factor:expr),+) => {{
        let mut bits = Vec::new();
        $(bits.extend(
            $factor
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .map(|v: &f64| v.to_bits()),
        );)+
        bits
    }};
}

/// After a cold call, one more call does `spaces` complete-structure hits, no
/// sector-structure lookup, and returns bit-identical data.
fn assert_one_lookup_per_space<T>(
    engine: &mut dyn Engine,
    label: &str,
    spaces: u64,
    call: impl Fn(&mut dyn Engine) -> T,
    bits: impl Fn(&mut dyn Engine, &T) -> Vec<u64>,
) {
    let cold = call(engine);
    let cold = bits(engine, &cold);
    let sector = info(StructureCacheKind::SectorStructure);
    let complete = info(StructureCacheKind::DegeneracyStructure);
    let warm = call(engine);
    let sector_after = info(StructureCacheKind::SectorStructure);
    let complete_after = info(StructureCacheKind::DegeneracyStructure);
    assert_eq!(bits(engine, &warm), cold, "{label}: warm result differs");
    assert_eq!(
        (
            sector_after.hits() - sector.hits(),
            sector_after.misses() - sector.misses()
        ),
        (0, 0),
        "{label}: warm sector-structure lookups"
    );
    assert_eq!(
        (
            complete_after.hits() - complete.hits(),
            complete_after.misses() - complete.misses()
        ),
        (spaces, 0),
        "{label}: warm complete-structure (hits, misses)"
    );
}

macro_rules! warm_derived {
    ($engine:ident, $label:literal, $leg:expr, $eigh_spaces:expr) => {{
        let runtime = host_runtime();
        let leg = $leg;
        let a = TensorMap::<_, f64>::rand_with_seed($engine, &runtime, [&leg, &leg], [&leg], 1)
            .unwrap();
        let b = TensorMap::<_, f64>::rand_with_seed($engine, &runtime, [&leg], [&leg, &leg], 2)
            .unwrap();
        let t =
            TensorMap::<_, f64>::rand_with_seed($engine, &runtime, [&leg, &leg], [&leg, &leg], 3)
                .unwrap();
        let t_adjoint = &t.adjoint($engine).unwrap().materialize($engine).unwrap();
        let h = t.axpby($engine, 1.0.into(), t_adjoint, 1.0.into()).unwrap();
        let (rows, cols) = (&[0, 1][..], &[2, 3][..]);
        assert_one_lookup_per_space(
            $engine,
            concat!($label, " contract"),
            1,
            |engine| a.compose(engine, &b).unwrap(),
            |engine, c| bits!(engine, c),
        );
        assert_one_lookup_per_space(
            $engine,
            concat!($label, " svd"),
            3,
            |engine| t.svd_compact(engine, rows, cols).unwrap(),
            |engine, Svd { u, s, vh }| bits!(engine, u, s, vh),
        );
        assert_one_lookup_per_space(
            $engine,
            concat!($label, " qr"),
            2,
            |engine| t.qr_compact(engine, rows, cols).unwrap(),
            |engine, Qr { q, r }| bits!(engine, q, r),
        );
        assert_one_lookup_per_space(
            $engine,
            concat!($label, " eigh"),
            $eigh_spaces,
            |engine| {
                h.eigh_full(engine, rows, cols, HermitianTol::DEFAULT)
                    .unwrap()
            },
            |engine, Eigh { d, v }| bits!(engine, d, v),
        );
    }};
}

#[test]
fn warm_derived_spaces_skip_the_sector_structure() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    symtn::cache::clear();
    warm_derived!(
        engine,
        "U1",
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(-1, 1), (0, 2), (1, 2)].map(|(q, n)| (U1Irrep::new(q), n)),
        )
        .unwrap(),
        3
    );
    warm_derived!(
        engine,
        "SU2",
        GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [(0, 2), (1, 2), (2, 1)].map(|(twice, n)| (SU2Irrep::from_twice_spin(twice), n)),
        )
        .unwrap(),
        3
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_derived_spaces_skip_the_sector_structure() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    symtn::cache::clear();
    let provider = Arc::new(
        symtn::sector::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap())
            .unwrap(),
    );
    warm_derived!(
        engine,
        "SU3",
        GradedSpace::try_new(provider, [(vec![1, 1], 2), (vec![0, 0], 1)]).unwrap(),
        // Checked Generic eigh derives two output spaces.
        2
    );
}
