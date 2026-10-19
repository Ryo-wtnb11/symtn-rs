//! Trace permutation columns through cache 4 (#2072): multiplicity-free
//! terms and outputs against the per-pair oracle, cold and warm; per-group
//! hits and misses, key splits and presence.

use super::*;
use crate::probe::{take_trace_column_activity, CoefficientGroupActivity};
use symtn_core::{
    clear_structure_caches, structure_cache_info, FermionParityFusionRule, StructureCacheKind,
};

type FpSu2Rule = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

fn real(i: usize) -> f64 {
    ((i * 7919 + 13) % 1009) as f64 / 37.0 - 13.0
}

fn complex(i: usize) -> Complex64 {
    Complex64::new(real(i), real(i + 503) / 3.0)
}

/// The materialized adjoint of `src` with payload `data`: TensorKit's
/// `block(t', (f2, f1)) = permutedims(conj(block(t, (f1, f2))))`, codomain
/// and domain swapped without dualizing.
fn materialized_adjoint<R, T>(
    provider: &Arc<R>,
    src: &BoundDynamicFusionMapSpace<R>,
    data: &[T],
    conj: impl Fn(T) -> T,
) -> (BoundDynamicFusionMapSpace<R>, Vec<T>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    T: Copy + Default,
{
    let hom = src.space().homspace();
    let adjoint = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        FusionTreeHomSpace::new(hom.domain().clone(), hom.codomain().clone()),
    )
    .unwrap();
    let parent = src.space().structure();
    let structure = adjoint.space().structure();
    let mut out = vec![T::default(); adjoint.space().required_len().unwrap()];
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let key = expect_tree_key(block.key());
        let source = parent
            .fusion_tree_pair_block(&FusionTreePairKey::pair(
                key.domain_tree().clone(),
                key.codomain_tree().clone(),
            ))
            .unwrap();
        let split = key.codomain_tree().uncoupled().len();
        for_each_index(block.shape(), |index| {
            let parent_index = [&index[split..], &index[..split]].concat();
            let at = |offset: usize, strides: &[usize], index: &[usize]| {
                offset + index.iter().zip(strides).map(|(i, s)| i * s).sum::<usize>()
            };
            out[at(block.offset(), block.strides(), index)] =
                conj(data[at(source.offset(), source.strides(), &parent_index)]);
        });
    }
    (adjoint, out)
}

fn for_each_index(shape: &[usize], mut visit: impl FnMut(&[usize])) {
    let count = shape.iter().product::<usize>();
    let mut index = vec![0; shape.len()];
    for linear in 0..count {
        let mut rest = linear;
        for (slot, &extent) in index.iter_mut().zip(shape) {
            *slot = rest % extent;
            rest /= extent;
        }
        visit(&index);
    }
}

/// `Σ_terms coefficient · (partial trace of the source block)` into each
/// destination block, by explicit index loops: destination axis `k` reads
/// source axis `output[k]`, and each traced pair shares one index.
fn trace_data_oracle<T>(
    terms: &[(FusionTreePairKey, FusionTreePairKey, usize, usize, f64)],
    dst: &BlockStructure,
    dst_len: usize,
    src: &BlockStructure,
    data: &[T],
    (output, lhs, rhs): (&[usize], &[usize], &[usize]),
) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<f64, Output = T>,
{
    let mut out = vec![T::default(); dst_len];
    for (_, _, dst_block, src_block, coefficient) in terms {
        let (d, s) = (
            dst.block(*dst_block).unwrap(),
            src.block(*src_block).unwrap(),
        );
        let traced = lhs.iter().map(|&axis| s.shape()[axis]).collect::<Vec<_>>();
        for_each_index(d.shape(), |open| {
            let mut src_index = vec![0; s.shape().len()];
            for (k, &axis) in output.iter().enumerate() {
                src_index[axis] = open[k];
            }
            let mut sum = T::default();
            for_each_index(&traced, |shared| {
                for (i, (&l, &r)) in lhs.iter().zip(rhs).enumerate() {
                    src_index[l] = shared[i];
                    src_index[r] = shared[i];
                }
                let at = s.offset()
                    + src_index
                        .iter()
                        .zip(s.strides())
                        .map(|(i, stride)| i * stride)
                        .sum::<usize>();
                sum += data[at];
            });
            let at = d.offset()
                + open
                    .iter()
                    .zip(d.strides())
                    .map(|(i, stride)| i * stride)
                    .sum::<usize>();
            out[at] += sum * *coefficient;
        });
    }
    out
}

/// One multiplicity-free trace, cold (after a cache reset) then warm: the
/// compiled terms equal [`scalar_trace_term_oracle`] (unless `conjugate`,
/// whose terms are over the parent layout), and the f64 and c64 outputs
/// equal [`trace_data_oracle`] over the source, or over its materialized
/// adjoint when `conjugate`. Warm equals cold bit for bit.
fn check_mf_trace<R>(
    name: &str,
    provider: Arc<R>,
    hom: FusionTreeHomSpace,
    (output, lhs, rhs): (&[usize], &[usize], &[usize]),
    conjugate: bool,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + symtn_core::CheckedFusionAlgebra,
{
    let axes = TensorTraceAxisSpec::new_with_conjugation(output, lhs, rhs, conjugate);
    let nout = output.len().min(1);
    let trace = MfTrace::new(Arc::clone(&provider), hom, axes, nout);
    let len = trace.src.space().required_len().unwrap();
    let real_data = (0..len).map(real).collect::<Vec<_>>();
    let complex_data = (0..len).map(complex).collect::<Vec<_>>();
    // A conjugating spec names the stored tensor's axes; the trace acts on
    // its adjoint, whose axis for stored axis `a` is TensorKit's
    // `adjointtensorindex`: codomain axis `a` becomes `nin + a`, domain axis
    // `a` becomes `a - nout`.
    let (stored_nout, stored_nin) = (trace.src.space().nout(), trace.src.space().nin());
    let logical_axis = |axis: &usize| match (conjugate, *axis < stored_nout) {
        (false, _) => *axis,
        (true, true) => stored_nin + axis,
        (true, false) => axis - stored_nout,
    };
    let map = |axes: &[usize]| axes.iter().map(logical_axis).collect::<Vec<_>>();
    let (output, lhs, rhs) = (map(output), map(lhs), map(rhs));
    let (logical, real_logical, complex_logical) = if conjugate {
        let (space, real_adjoint) = materialized_adjoint(&provider, &trace.src, &real_data, |x| x);
        let (_, complex_adjoint) =
            materialized_adjoint(&provider, &trace.src, &complex_data, |x| x.conj());
        (space, real_adjoint, complex_adjoint)
    } else {
        (trace.src.clone(), real_data.clone(), complex_data.clone())
    };
    // The destination the conjugating trace selects is the one a plain trace
    // of the materialized adjoint selects.
    let logical_dst = crate::tensortrace_fusion_dyn_preflight_checked(
        &logical,
        TensorTraceAxisSpec::new(&output, &lhs, &rhs),
        nout,
    )
    .unwrap()
    .into_selected_homspace();
    assert_eq!(&logical_dst, trace.dst.space().homspace(), "{name}");
    let oracle = scalar_trace_term_oracle(
        provider.as_ref(),
        trace.dst.space().structure(),
        logical.space().structure(),
        &output,
        &lhs,
        &rhs,
        nout,
    );
    assert!(!oracle.is_empty(), "{name}");
    let dst_len = trace.dst.space().required_len().unwrap();
    let (dst, src) = (trace.dst.space().structure(), logical.space().structure());
    let axes_triple = (&output[..], &lhs[..], &rhs[..]);
    let want_real = trace_data_oracle(&oracle, dst, dst_len, src, &real_logical, axes_triple);
    let want_complex = trace_data_oracle(&oracle, dst, dst_len, src, &complex_logical, axes_triple);
    let run = || {
        let structure = trace.compile(axes);
        if !conjugate {
            assert_eq!(structure.terms().len(), oracle.len(), "{name}");
            for (actual, (dst_key, src_key, dst_block, src_block, coefficient)) in
                structure.terms().iter().zip(&oracle)
            {
                assert_eq!(
                    (
                        actual.dst_key(),
                        actual.src_key(),
                        actual.dst_block(),
                        actual.src_block()
                    ),
                    (dst_key, src_key, *dst_block, *src_block),
                    "{name}"
                );
                assert!(
                    (actual.coefficient() - coefficient).abs()
                        <= 1.0e-12 * (1.0 + coefficient.abs()),
                    "{name}"
                );
            }
        }
        let real_out = crate::tensortrace_fusion_dyn_owned_checked(
            &trace.dst, &trace.src, &real_data, axes, 1.0,
        )
        .unwrap();
        let complex_out = crate::tensortrace_fusion_dyn_owned_checked(
            &trace.dst,
            &trace.src,
            &complex_data,
            axes,
            Complex64::new(1.0, 0.0),
        )
        .unwrap();
        for (index, (got, want)) in real_out.iter().zip(&want_real).enumerate() {
            assert!(
                (got - want).abs() <= 1e-11 * (1.0 + want.abs()),
                "{name} f64 [{index}]: {got} vs {want}"
            );
        }
        for (index, (got, want)) in complex_out.iter().zip(&want_complex).enumerate() {
            assert!(
                (got - want).norm() <= 1e-11 * (1.0 + want.norm()),
                "{name} c64 [{index}]: {got} vs {want}"
            );
        }
        assert!(want_real.iter().any(|value| *value != 0.0), "{name}");
        (
            real_out
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            complex_out
                .iter()
                .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
                .collect::<Vec<_>>(),
        )
    };
    let _guard = cache_lock();
    clear_structure_caches();
    let cold = run();
    let warm = run();
    assert_eq!(cold, warm, "{name}: warm differs from cold");
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [(0, 2), (1, 1), (2, 2)]
            .map(|(spin, degeneracy)| (SU2Irrep::from_twice_spin(spin).sector_id(), degeneracy)),
        false,
    )
}

fn su2_hom() -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(
        FusionProductSpace::new([su2_leg(), su2_leg()]),
        FusionProductSpace::new([su2_leg(), su2_leg()]),
    )
}

fn fp_su2_hom() -> FusionTreeHomSpace {
    let rule = FpSu2Rule::default();
    let leg = |dual| {
        SectorLeg::new(
            [(0, 0, 1), (1, 1, 2), (1, 2, 1)].map(|(parity, spin, degeneracy)| {
                (
                    rule.encode_component_ids(
                        SectorId::new(parity),
                        SU2Irrep::from_twice_spin(spin).sector_id(),
                    ),
                    degeneracy,
                )
            }),
            dual,
        )
    };
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    )
}

fn u1_hom() -> FusionTreeHomSpace {
    let leg = || {
        SectorLeg::new(
            [(-1, 1), (0, 2), (1, 1)]
                .map(|(charge, degeneracy)| (U1Irrep::new(charge).sector_id(), degeneracy)),
            false,
        )
    };
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    )
}

/// What: multiplicity-free trace terms and outputs (one `(p…, q…)` braid,
/// the identity permutation, the lazy adjoint, two pairs) equal the per-pair
/// oracle and its explicit index-loop evaluation for SU(2), the fermionic
/// fZ2⊠SU(2) product (twist factor) and U(1) (Unique), cold and warm.
#[test]
fn mf_trace_terms_and_outputs_match_the_per_pair_oracle_cold_and_warm() {
    let open: (&[usize], &[usize], &[usize]) = (&[1, 3], &[0], &[2]);
    let identity: (&[usize], &[usize], &[usize]) = (&[0, 2], &[1], &[3]);
    let both: (&[usize], &[usize], &[usize]) = (&[], &[0, 1], &[2, 3]);
    let su2 = || Arc::new(SU2FusionRule);
    let fp = || Arc::new(FpSu2Rule::default());
    let u1 = || Arc::new(U1FusionRule);
    check_mf_trace("su2 open", su2(), su2_hom(), open, false);
    check_mf_trace("su2 identity", su2(), su2_hom(), identity, false);
    check_mf_trace("su2 adjoint", su2(), su2_hom(), open, true);
    check_mf_trace("su2 both", su2(), su2_hom(), both, false);
    check_mf_trace("fp⊠su2 open", fp(), fp_su2_hom(), open, false);
    check_mf_trace("fp⊠su2 identity", fp(), fp_su2_hom(), identity, false);
    check_mf_trace("fp⊠su2 adjoint", fp(), fp_su2_hom(), open, true);
    check_mf_trace("fp⊠su2 both", fp(), fp_su2_hom(), both, false);
    check_mf_trace("u1 open", u1(), u1_hom(), open, false);
    check_mf_trace("u1 adjoint", u1(), u1_hom(), open, true);
}

/// One multiplicity-free trace: its bound source and selected destination.
struct MfTrace<R> {
    src: BoundDynamicFusionMapSpace<R>,
    dst: BoundDynamicFusionMapSpace<R>,
}

impl<R> MfTrace<R>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + symtn_core::CheckedFusionAlgebra,
{
    fn new(
        provider: Arc<R>,
        hom: FusionTreeHomSpace,
        axes: TensorTraceAxisSpec<'_>,
        nout: usize,
    ) -> Self {
        let src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            hom,
        )
        .unwrap();
        let dst_hom = crate::tensortrace_fusion_dyn_preflight_checked(&src, axes, nout)
            .unwrap()
            .into_selected_homspace();
        let dst =
            BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(provider, dst_hom)
                .unwrap();
        Self { src, dst }
    }

    fn compile(&self, axes: TensorTraceAxisSpec<'_>) -> TensorTraceFusionStructure<f64> {
        TensorTraceFusionStructure::compile_fusion_dyn_checked(&self.dst, &self.src, axes).unwrap()
    }

    fn groups(&self) -> usize {
        self.src.space().structure().fusion_tree_group_slice().len()
    }
}

type TermBits = Vec<(FusionTreePairKey, FusionTreePairKey, usize, usize, u64)>;

fn term_bits(structure: &TensorTraceFusionStructure<f64>) -> TermBits {
    structure
        .terms()
        .iter()
        .map(|term| {
            (
                term.dst_key().clone(),
                term.src_key().clone(),
                term.dst_block(),
                term.src_block(),
                term.coefficient().to_bits(),
            )
        })
        .collect()
}

fn activity(hits: usize, misses: usize, publications: usize) -> CoefficientGroupActivity {
    CoefficientGroupActivity {
        hits,
        misses,
        publications,
    }
}

fn cache_lock() -> std::sync::MutexGuard<'static, ()> {
    crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

/// SU(2) `V ⊗ V ← V ⊗ V̄`, `V = 0 ⊕ 1 ⊕ 2` (spins), traced over the two
/// domain legs: its permutation keeps structurally present exact zeros.
fn su2_present_zero_hom(degeneracy: usize) -> FusionTreeHomSpace {
    let leg = |dual| {
        SectorLeg::new(
            [0, 2, 4].map(|spin| (SU2Irrep::from_twice_spin(spin).sector_id(), degeneracy)),
            dual,
        )
    };
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(false)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    )
}

const PRESENT_ZERO_OUTPUT: [usize; 2] = [0, 1];
const PRESENT_ZERO_LHS: [usize; 1] = [2];
const PRESENT_ZERO_RHS: [usize; 1] = [3];

fn present_zero_axes() -> TensorTraceAxisSpec<'static> {
    TensorTraceAxisSpec::new(&PRESENT_ZERO_OUTPUT, &PRESENT_ZERO_LHS, &PRESENT_ZERO_RHS)
}

/// What: a Simple trace resolves each fusion group once per call from
/// cache 4: cold misses and publishes every group, warm hits every group
/// with identical term bits. Another trace pair, the lazy adjoint of the
/// same parent and a transform-scope entry of the same permutation are
/// distinct entries; a degeneracy-only change and another instance of the
/// same rule hit; Unique fusion makes no cache-4 activity (TensorKit
/// `NoCache`).
#[test]
fn trace_groups_resolve_through_cache4_per_group() {
    // Exact hit/miss counts: alone in a child process, so no sibling test
    // publishes or clears these shared SU(2) groups meanwhile.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_2072_ISOLATED",
        "tests::tensortrace::trace_cache4::trace_groups_resolve_through_cache4_per_group",
    ) {
        return;
    }
    clear_structure_caches();
    take_trace_column_activity();
    let open = TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]);
    let su2 = MfTrace::new(Arc::new(SU2FusionRule), su2_hom(), open, 1);
    let groups = su2.groups();
    assert!(groups > 1);

    let cold = su2.compile(open);
    assert_eq!(take_trace_column_activity(), activity(0, groups, groups));
    let warm = su2.compile(open);
    assert_eq!(take_trace_column_activity(), activity(groups, 0, 0));
    assert_eq!(term_bits(&cold), term_bits(&warm));

    // Key splits: another `(p…, q…)` and the lazy adjoint of one parent.
    let identity = TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]);
    MfTrace::new(Arc::new(SU2FusionRule), su2_hom(), identity, 1).compile(identity);
    assert_eq!(take_trace_column_activity(), activity(0, groups, groups));
    let adjoint = TensorTraceAxisSpec::new_with_conjugation(&[1, 3], &[0], &[2], true);
    MfTrace::new(Arc::new(SU2FusionRule), su2_hom(), adjoint, 1).compile(adjoint);
    assert_eq!(take_trace_column_activity(), activity(0, groups, groups));

    // Degeneracies are not keyed.
    let scaled_leg = || {
        SectorLeg::new(
            [(0, 3), (1, 2), (2, 1)].map(|(spin, degeneracy)| {
                (SU2Irrep::from_twice_spin(spin).sector_id(), degeneracy)
            }),
            false,
        )
    };
    let scaled = MfTrace::new(
        Arc::new(SU2FusionRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([scaled_leg(), scaled_leg()]),
            FusionProductSpace::new([scaled_leg(), scaled_leg()]),
        ),
        open,
        1,
    );
    scaled.compile(open);
    assert_eq!(take_trace_column_activity(), activity(groups, 0, 0));

    // Two instances of one rule share entries (`RuleIdentity`).
    let fp_groups = MfTrace::new(Arc::new(FpSu2Rule::default()), fp_su2_hom(), open, 1).groups();
    MfTrace::new(Arc::new(FpSu2Rule::default()), fp_su2_hom(), open, 1).compile(open);
    assert_eq!(
        take_trace_column_activity(),
        activity(0, fp_groups, fp_groups)
    );
    MfTrace::new(Arc::new(FpSu2Rule::default()), fp_su2_hom(), open, 1).compile(open);
    assert_eq!(take_trace_column_activity(), activity(fp_groups, 0, 0));

    let u1 = MfTrace::new(Arc::new(U1FusionRule), u1_hom(), open, 1);
    u1.compile(open);
    u1.compile(open);
    assert_eq!(take_trace_column_activity(), activity(0, 0, 0));
}

/// What: a transform and a trace of the same permutation on the same source
/// group are separate cache-4 entries (scope `TraceTerms`); neither reads
/// the other's value, and each still hits its own entry afterwards.
#[test]
fn trace_and_transform_of_one_permutation_do_not_share_entries() {
    // Exact hit/miss counts: alone in a child process, so no sibling test
    // publishes or clears these shared SU(2) groups meanwhile.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_2072_ISOLATED",
        "tests::tensortrace::trace_cache4::trace_and_transform_of_one_permutation_do_not_share_entries",
    ) {
        return;
    }
    clear_structure_caches();
    take_trace_column_activity();
    crate::probe::take_coefficient_group_activity();
    let open = TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]);
    // The trace permutes `[1, 0] ← [3, 2]`; on `V² ← V²` its output space is
    // the source space.
    let operation = crate::TreeTransformOperation::permute([1, 0], [3, 2]);
    let structure = |degeneracy| {
        Arc::clone(
            BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
                Arc::new(SU2FusionRule),
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([su2_leg_with(degeneracy), su2_leg_with(degeneracy)]),
                    FusionProductSpace::new([su2_leg_with(degeneracy), su2_leg_with(degeneracy)]),
                ),
            )
            .unwrap()
            .space()
            .structure(),
        )
    };
    let mut planning = crate::tree_transform::TreeTransformPlanning::default();
    let mut transform = |degeneracy| {
        let layout = structure(degeneracy);
        planning
            .resolve_tree_pair(&SU2FusionRule, &operation, &layout, &layout, false)
            .unwrap()
    };
    let su2 = MfTrace::new(Arc::new(SU2FusionRule), su2_hom(), open, 1);
    let groups = su2.groups();
    let before = transform(41);
    let transform_cold = crate::probe::take_coefficient_group_activity();
    assert_eq!(transform_cold.misses, groups);
    let cold = su2.compile(open);
    assert_eq!(take_trace_column_activity(), activity(0, groups, groups));
    let after = transform(43);
    assert_eq!(
        crate::probe::take_coefficient_group_activity(),
        activity(groups, 0, 0)
    );
    let warm = su2.compile(open);
    assert_eq!(take_trace_column_activity(), activity(groups, 0, 0));
    assert_eq!(term_bits(&cold), term_bits(&warm));
    let bits = |structure: &crate::TreeTransformStructure<f64>| {
        let mut coefficients = Vec::new();
        structure.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
            .iter()
            .map(|value: &f64| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&before), bits(&after));
}

/// What (#2149): the open codomain rank is keyed. A one-pair trace and a
/// pair-free trace share the `(p…, q…)` permutation `[1, 0] ← [3, 2]` but
/// split it differently, so the pair-free one misses after the one-pair one
/// is resident, and its terms equal its own compile on an empty cache.
#[test]
fn trace_entries_key_the_open_codomain_rank() {
    // Exact hit/miss counts: alone in a child process, so no sibling test
    // publishes or clears these shared SU(2) groups meanwhile.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_2072_ISOLATED",
        "tests::tensortrace::trace_cache4::trace_entries_key_the_open_codomain_rank",
    ) {
        return;
    }
    let one_pair = TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]);
    let pair_free = TensorTraceAxisSpec::new(&[1, 0, 3, 2], &[], &[]);
    let traced = MfTrace::new(Arc::new(SU2FusionRule), su2_hom(), one_pair, 1);
    let permuted = MfTrace::new(Arc::new(SU2FusionRule), su2_hom(), pair_free, 2);
    let groups = permuted.groups();
    clear_structure_caches();
    let alone = term_bits(&permuted.compile(pair_free));
    assert!(!alone.is_empty());
    clear_structure_caches();
    traced.compile(one_pair);
    take_trace_column_activity();
    let after = permuted.compile(pair_free);
    assert_eq!(take_trace_column_activity(), activity(0, groups, groups));
    assert_eq!(term_bits(&after), alone);
}

fn su2_leg_with(degeneracy: usize) -> SectorLeg {
    SectorLeg::new(
        [0, 1, 2].map(|spin| (SU2Irrep::from_twice_spin(spin).sector_id(), degeneracy)),
        false,
    )
}

/// What: `symtn::cache::stats()`' cache-4 entries rise by one per Simple
/// trace group.
#[test]
fn trace_groups_are_cache4_entries() {
    if crate::test_support::run_isolated_or_return(
        "SYMTN_2072_ISOLATED",
        "tests::tensortrace::trace_cache4::trace_groups_are_cache4_entries",
    ) {
        return;
    }
    let open = TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]);
    let su2 = MfTrace::new(Arc::new(SU2FusionRule), su2_hom(), open, 1);
    clear_structure_caches();
    let entries = || structure_cache_info(StructureCacheKind::TreeTransformCoefficients).entries();
    assert_eq!(entries(), 0);
    su2.compile(open);
    assert_eq!(entries(), su2.groups());
    su2.compile(open);
    assert_eq!(entries(), su2.groups());
}

/// What: the compiled multiplicity-free terms, cold and warm, equal an
/// independent per-pair composition: each source permuted alone by
/// `multiplicity_free_permute_tree_pair`, split, matched, scaled by
/// `dim(c)/dim(first)·Π twist(non-dual)` (`scalar_trace_term_oracle`), for
/// SU(2) (with present zeros) and the fermionic fZ2⊠SU(2) product (twist).
#[test]
fn mf_trace_terms_match_per_pair_composition_cold_and_warm() {
    let _guard = cache_lock();
    clear_structure_caches();
    fn check<R>(
        provider: Arc<R>,
        hom: FusionTreeHomSpace,
        output: &[usize],
        lhs: &[usize],
        rhs: &[usize],
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + symtn_core::CheckedFusionAlgebra,
    {
        let axes = TensorTraceAxisSpec::new(output, lhs, rhs);
        let trace = MfTrace::new(Arc::clone(&provider), hom, axes, 1);
        let oracle = scalar_trace_term_oracle(
            provider.as_ref(),
            trace.dst.space().structure(),
            trace.src.space().structure(),
            output,
            lhs,
            rhs,
            1,
        );
        for _cold_then_warm in 0..2 {
            let structure = trace.compile(axes);
            assert_eq!(structure.terms().len(), oracle.len());
            for (actual, (dst_key, src_key, dst_block, src_block, coefficient)) in
                structure.terms().iter().zip(&oracle)
            {
                assert_eq!(
                    (
                        actual.dst_key(),
                        actual.src_key(),
                        actual.dst_block(),
                        actual.src_block()
                    ),
                    (dst_key, src_key, *dst_block, *src_block)
                );
                assert!(
                    (actual.coefficient() - coefficient).abs()
                        <= 1.0e-12 * (1.0 + coefficient.abs())
                );
            }
        }
    }
    check(Arc::new(SU2FusionRule), su2_hom(), &[1, 3], &[0], &[2]);
    check(
        Arc::new(SU2FusionRule),
        su2_present_zero_hom(1),
        &PRESENT_ZERO_OUTPUT,
        &PRESENT_ZERO_LHS,
        &PRESENT_ZERO_RHS,
    );
    check(
        Arc::new(FpSu2Rule::default()),
        fp_su2_hom(),
        &[1, 3],
        &[0],
        &[2],
    );
    check(
        Arc::new(FpSu2Rule::default()),
        fp_su2_hom(),
        &[0, 2],
        &[1],
        &[3],
    );
}

/// What: present zero permutation entries stay terms after a cache hit
/// (term bits equal cold), and a NaN source block reaches the destination
/// of such a term warm as cold, bit for bit. An absent entry makes no term:
/// the term lists equal the per-pair oracle above. (In this fixture every
/// zero term shares its block pair with a nonzero term, so the output alone
/// cannot isolate the zero term's NaN.)
#[test]
fn present_zero_entry_propagates_nan_warm_as_cold() {
    // Exact hit/miss counts: alone in a child process, so no sibling test
    // publishes or clears these shared SU(2) groups meanwhile.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_2072_ISOLATED",
        "tests::tensortrace::trace_cache4::present_zero_entry_propagates_nan_warm_as_cold",
    ) {
        return;
    }
    clear_structure_caches();
    let axes = present_zero_axes();
    let trace = MfTrace::new(Arc::new(SU2FusionRule), su2_present_zero_hom(2), axes, 1);
    let cold = trace.compile(axes);
    take_trace_column_activity();
    let warm = trace.compile(axes);
    assert_eq!(take_trace_column_activity().misses, 0);
    assert_eq!(term_bits(&cold), term_bits(&warm));
    let zero = warm
        .terms()
        .iter()
        .find(|term| *term.coefficient() == 0.0)
        .expect("fixture keeps a present zero entry");
    let src_structure = trace.src.space().structure();
    let block = src_structure.block(zero.src_block()).unwrap();
    let mut data = (0..trace.src.space().required_len().unwrap())
        .map(real)
        .collect::<Vec<_>>();
    data[block.offset()..block.storage_end_exclusive().unwrap()].fill(f64::NAN);
    let run = || {
        crate::tensortrace_fusion_dyn_owned_checked(&trace.dst, &trace.src, &data, axes, 1.0)
            .unwrap()
    };
    // The compiles above left the groups resident: reset so this run is cold.
    let cold_out = {
        let _guard = cache_lock();
        clear_structure_caches();
        take_trace_column_activity();
        let out = run();
        assert_eq!(take_trace_column_activity().hits, 0);
        out
    };
    let warm_out = run();
    assert_eq!(
        cold_out
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        warm_out
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    let dst_block = trace
        .dst
        .space()
        .structure()
        .block(zero.dst_block())
        .unwrap();
    assert!(
        warm_out[dst_block.offset()..dst_block.storage_end_exclusive().unwrap()]
            .iter()
            .any(|value| value.is_nan())
    );
}
