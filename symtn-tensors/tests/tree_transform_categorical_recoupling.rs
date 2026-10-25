//! CPU-only half of the categorical recoupling evidence for the device
//! tree-transform executor (issue #1310).
//!
//! The device suite (`cuda_tree_transform_categorical.rs`, `#[ignore]`) replays
//! provider-compiled SU(2) and fZ2 x SU(2) structures and compares them with
//! the oracle in `categorical_recoupling`. That is only evidence if the oracle
//! itself is right, so this file — which needs no GPU and therefore runs in CI —
//! pins the oracle against the *host* executor replaying the same compiled
//! structures, and proves the fixtures actually contain recoupling.

use crate::categorical_recoupling;

use categorical_recoupling::{
    alphas, assert_close, compile, expected, expected_scaled, fermionic_space, fermionic_su2_rule,
    fixtures, four_leg_channels, host_replay, host_replay_scaled, non_symmetric_fixture, positions,
    su2_space, Compiled, Side, TestScalar,
};
use num_complex::Complex64;
use symtn_core::{FermionParityFusionRule, SU2FusionRule};
use symtn_operations::TreeTransformBlock;
use symtn_tensors::TreeTransformOperation;

fn check<T>(fixture: &Compiled)
where
    T: TestScalar
        + symtn_operations::TreeTransformScalar
        + symtn_operations::RecouplingCoefficientAction<f64>
        + symtn_operations::DenseBlockScalar,
{
    let source = fixture.source::<T>();
    let destination: Vec<T> = (0..fixture.len())
        .map(|index| T::from_parts(-3.0 - index as f64, 0.5))
        .collect();
    for overwrite in [true, false] {
        for alpha in alphas::<T>() {
            let what = format!(
                "{} / {} / overwrite = {overwrite} / alpha = {alpha:?}",
                fixture.name,
                T::NAME
            );
            assert_close(
                &host_replay_scaled(fixture, &source, &destination, overwrite, alpha),
                &expected_scaled(fixture, &source, &destination, overwrite, alpha),
                &what,
            );
        }
    }
}

#[test]
fn the_caller_scale_multiplies_a_recoupled_block_once() {
    // Negative control for where alpha enters provider-compiled recoupling: the
    // host applies it once, at the scatter. Applying it at the pack as well
    // would give `alpha^2 * (U x)` — which is exactly what the oracle computes
    // for `alpha * alpha` — so the two must differ and the host must land on
    // the single application.
    let fixture = non_symmetric_fixture();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.len()];
    let alpha = -2.5_f64;

    let once = expected_scaled(&fixture, &source, &destination, true, alpha);
    let twice = expected_scaled(&fixture, &source, &destination, true, alpha * alpha);
    assert_ne!(once, twice, "alpha applied twice must change the result");
    assert_close(
        &host_replay_scaled(&fixture, &source, &destination, true, alpha),
        &once,
        "alpha applied once at the scatter",
    );
}

#[test]
fn the_categorical_fixtures_really_recouple() {
    // What: every fixture the device suite replays contains a genuine Multi
    // block, so "device agrees with the oracle" is a statement about the
    // pack -> U^T -> scatter path and cannot be satisfied by a structure that
    // happens to be all Single blocks. At least one of them must additionally
    // have a *non-symmetric* recoupling matrix — the two-channel SU(2) F move
    // is its own transpose, which is real 6j data but blind to a reversed
    // orientation, so a suite built only from those would prove nothing about
    // the GEMM orientation.
    let fixtures = fixtures();
    assert!(!fixtures.is_empty());
    for fixture in &fixtures {
        let shapes = fixture.multi_shapes();
        assert!(
            !shapes.is_empty(),
            "{} compiled without a recoupling block",
            fixture.name
        );
        assert!(
            shapes.iter().any(|(src, dst)| *src > 1 || *dst > 1),
            "{}: degenerate recoupling shapes {shapes:?}",
            fixture.name
        );
    }
    assert!(
        fixtures
            .iter()
            .any(|fixture| fixture.has_non_symmetric_matrix()),
        "no fixture can distinguish U from its transpose"
    );
    assert!(
        fixtures.iter().any(|fixture| {
            fixture.name.starts_with("fermionic") && fixture.has_non_symmetric_matrix()
        }),
        "no fermionic fixture with a non-symmetric recoupling matrix"
    );
    assert!(
        fixtures.iter().any(|fixture| fixture.conjugate),
        "no conjugated-source fixture"
    );
}

#[test]
fn the_categorical_oracle_agrees_with_host_replay() {
    // What: SU(2) and fZ2 x SU(2) permutes, braids and planar transposes —
    // compiled by the real providers, with irrational 6j entries and, for the
    // fermionic rule, fermionic signs inside U — mean element for element the
    // plain weighted sum the oracle computes, in both payload dtypes, both
    // destination modes and with a conjugated source.
    for fixture in fixtures() {
        check::<f64>(&fixture);
        check::<Complex64>(&fixture);
    }
}

/// The same walk as the oracle, but applying `Uᵀ` where it applies `U`.
///
/// Deliberately a second, explicit loop rather than a flag on the oracle, so
/// the oracle under test stays the plain one.
fn transposed_walk(fixture: &Compiled, source: &[f64]) -> Vec<f64> {
    let structure = &fixture.structure;
    let coefficients = structure.gathered_coefficients();
    let mut transposed = vec![0.0_f64; fixture.len()];
    for block in structure.blocks() {
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            coefficient_start,
            ..
        } = *block
        else {
            continue;
        };
        assert_eq!(src_count, dst_count, "the control needs a square U");
        for dst_index in 0..dst_count {
            for src_index in 0..src_count {
                // Swapped indices: Uᵀ.
                let coefficient =
                    coefficients[coefficient_start + src_index * src_count + dst_index];
                let dst = positions(fixture, dst_layout_start + dst_index, Side::Destination);
                let src = positions(fixture, src_layout_start + src_index, Side::Source);
                for (dst_position, src_position) in dst.into_iter().zip(src) {
                    transposed[dst_position] += coefficient * source[src_position];
                }
            }
        }
    }
    transposed
}

#[test]
fn the_categorical_oracle_distinguishes_u_from_its_transpose() {
    // Negative control for the GEMM orientation on *provider* data: transposing
    // the recoupling matrix of a compiled fixture changes the oracle's answer,
    // and the host follows the untransposed one. Without this, "host agrees
    // with the oracle" would hold under a reversed orientation too.
    let fixture = non_symmetric_fixture();
    let source = fixture.source::<f64>();
    let destination = vec![0.0_f64; fixture.len()];
    let untransposed = expected(&fixture, &source, &destination, true);

    assert_ne!(
        transposed_walk(&fixture, &source),
        untransposed,
        "the fixture's recoupling matrix is symmetric, so this control is blind"
    );
    assert_close(
        &host_replay(&fixture, &source, &destination, true),
        &untransposed,
        "su2_rank6_permute / overwrite",
    );
}

#[test]
fn a_fermionic_abelian_rule_compiles_to_single_blocks_only() {
    // Why the fermionic fixture is a *product* rule: fZ2 on its own is
    // `FusionStyleKind::Unique`, so every fusion tree of a key is forced and an
    // F move is 1x1. A fermionic abelian structure therefore exercises signs
    // but never the recoupling path — which is why the fermionic recoupling
    // evidence above uses fZ2 x SU(2), the one rule here that is both fermionic
    // and multi-channel. Pinned so a future reader does not "simplify" the
    // fixture back to plain fZ2 and silently lose the Multi coverage.
    let space = categorical_recoupling::fermion_parity_structure(2);
    let fixture = compile(
        "fz2_braid",
        &FermionParityFusionRule,
        TreeTransformOperation::braid([0, 2, 1, 3], [], 0..4, []),
        &space,
        false,
    );

    assert!(
        fixture.multi_shapes().is_empty(),
        "plain fZ2 unexpectedly recoupled: {:?}",
        fixture.multi_shapes()
    );
    // It still replays, and the oracle still describes it: the Single path is
    // the one carrying the fermionic sign here.
    check::<f64>(&fixture);
    check::<Complex64>(&fixture);
}

#[test]
fn the_fermionic_recoupling_matrix_differs_from_the_bosonic_one() {
    // What: the fZ2 x SU(2) fixtures are not the SU(2) ones relabelled — their
    // recoupling entries carry the fermionic signs — so the fermionic evidence
    // is about fermionic data and not a second copy of the bosonic case.
    let rule = fermionic_su2_rule();
    let legs = [1usize; 4];
    let channels = four_leg_channels();
    let swap = [0usize, 2, 1, 3];
    let bosonic = compile(
        "su2_braid",
        &SU2FusionRule,
        TreeTransformOperation::braid(swap, [], 0..4, []),
        &su2_space(&legs, &channels, 2),
        false,
    );
    let fermionic = compile(
        "fermionic_su2_braid",
        &rule,
        TreeTransformOperation::braid(swap, [], 0..4, []),
        &fermionic_space(&rule, &legs, &channels, 2),
        false,
    );

    assert_ne!(
        bosonic.structure.gathered_coefficients(),
        fermionic.structure.gathered_coefficients(),
        "the fermionic rule produced the bosonic coefficients"
    );
}

/// Test view of the logical coefficient payload (an explicit copy).
trait GatheredCoefficients<T> {
    fn gathered_coefficients(&self) -> Vec<T>;
}

impl<T: Copy> GatheredCoefficients<T> for symtn_tensors::TreeTransformStructure<T> {
    fn gathered_coefficients(&self) -> Vec<T> {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }
}
