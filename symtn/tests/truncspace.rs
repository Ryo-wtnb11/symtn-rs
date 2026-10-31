//! `truncspace` — fixed per-sector truncation on the typed facade.
//!
//! TensorKit `TruncationSpace` (`src/factorizations/truncation.jl:261-269`)
//! reads the target rank of coupled sector `c` as `dim(strategy.space, c)` and
//! applies a plain `truncrank` inside that block, so the decision is a fixed
//! per-sector prefix count and nothing about the spectrum's magnitudes enters.
//!
//! The exact kept set of `Space` and `Space & Rank` is checked against the
//! hand selection in every `truncation_composition` sweep; this file keeps the
//! clamping and dual-flag boundaries.

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Error, Truncation};
use symtn::typed::{GradedSpace, Svd, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

/// A deterministic but structureless fill, so the singular values carry no
/// pattern a magnitude-driven policy could accidentally reproduce.
fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

fn typed_leg(entries: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        entries
            .iter()
            .map(|&(twice_spin, deg)| (SU2Irrep::from_twice_spin(twice_spin), deg)),
    )
    .unwrap()
}

fn typed_source(seed: u64) -> TensorMap<SU2FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let leg = typed_leg(&[(0, 4), (1, 4), (2, 4)]);
    let mut state = seed;
    TensorMap::from_subblock_fn(engine, &host_runtime(), [&leg], [&leg], move |_, _| {
        fill(&mut state)
    })
    .unwrap()
}

/// The truncated SVD's `u` and error: `svd_compact` -> `diagview` ->
/// `find_truncated` -> `restrict_leg`.
fn truncated_svd<R>(
    source: &TensorMap<R, f64>,
    truncation: &Truncation,
) -> Result<(TensorMap<R, f64>, f64), Error>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let Svd { u, s, .. } = source.svd_compact(engine, &[0], &[1])?;
    let found = s.domain()[0].find_truncated(&s.diagview(engine)?, truncation)?;
    Ok((
        u.restrict_leg(engine, &[(u.codomain_rank(), &found.selection)])?,
        found.error,
    ))
}

fn typed_u1_leg(entries: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        entries
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

#[test]
fn typed_truncspace_clamps_a_request_longer_than_the_spectrum() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let source = typed_source(0x5eed_1000);
    let greedy = typed_leg(&[(0, 99), (1, 99), (2, 99)]);
    let (clamped, clamped_error) =
        truncated_svd(&source, &Truncation::space(greedy.truncspace())).unwrap();
    let full = source.svd_compact(engine, &[0], &[1]).unwrap();
    let clamped_bond = &clamped.domain()[0];
    let full_bond = &full.u.domain()[0];

    assert_eq!(
        clamped_bond.sectors().unwrap(),
        full_bond.sectors().unwrap()
    );
    assert_eq!(clamped_bond.degeneracies(), full_bond.degeneracies());
    assert_eq!(clamped_error, 0.0);
}

#[test]
fn typed_truncspace_follows_stored_sectors_not_the_dual_flag() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let leg = typed_u1_leg(&[(-1, 3), (0, 3), (1, 3)]);
    let source = TensorMap::<U1FusionRule, f64>::rand_with_seed(
        engine,
        &host_runtime(),
        [&leg],
        [&leg],
        0x5eed_6000,
    )
    .unwrap();
    let bond = |space: &GradedSpace<U1FusionRule>| {
        truncated_svd(&source, &Truncation::space(space.truncspace()))
            .unwrap()
            .0
            .domain()[0]
            .clone()
    };

    // Content half: an asymmetric target and its dual carry mirrored charges,
    // so they must select mirrored bond spaces — the profile reads the dual's
    // rewritten ids, exactly as TensorKit's `dim(V', c)` does.
    let skewed = typed_u1_leg(&[(-1, 3), (0, 2), (1, 1)]);
    let skewed_dual = skewed.try_dual().unwrap();
    let skewed_bond = bond(&skewed);
    let skewed_dual_bond = bond(&skewed_dual);
    assert_eq!(skewed_bond.sectors().unwrap(), skewed.sectors().unwrap());
    assert_eq!(skewed_bond.degeneracies(), skewed.degeneracies());
    assert_eq!(
        skewed_dual_bond.sectors().unwrap(),
        skewed_dual.sectors().unwrap()
    );
    assert_eq!(skewed_dual_bond.degeneracies(), skewed_dual.degeneracies());
    assert_ne!(
        skewed_bond.degeneracies(),
        skewed_dual_bond.degeneracies(),
        "a non-self-dual target and its dual must not select the same bond"
    );

    // Flag half: a charge-symmetric target has the *same* stored `(id, deg)`
    // pairs as its dual and differs only in the flag, so the two profiles must
    // agree. This is the assertion that dies if the flag ever leaks in.
    let symmetric = typed_u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let symmetric_dual = symmetric.try_dual().unwrap();
    assert!(symmetric_dual.is_dual() && !symmetric.is_dual());
    assert_eq!(
        symmetric.sectors().unwrap(),
        symmetric_dual.sectors().unwrap()
    );
    assert_eq!(symmetric.degeneracies(), symmetric_dual.degeneracies());
    let symmetric_bond = bond(&symmetric);
    let symmetric_dual_bond = bond(&symmetric_dual);
    assert_eq!(
        symmetric_bond.sectors().unwrap(),
        symmetric_dual_bond.sectors().unwrap()
    );
    assert_eq!(
        symmetric_bond.degeneracies(),
        symmetric_dual_bond.degeneracies()
    );
}
