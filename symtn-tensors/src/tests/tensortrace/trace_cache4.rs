//! Trace permutation columns through cache 4 (#2072): multiplicity-free
//! output bits pinned to a revision without this cache, cold and warm; per-group
//! hits and misses, key splits, presence and the per-pair oracle.

use super::*;
use crate::probe::{take_trace_column_activity, CoefficientGroupActivity};
use symtn_core::{
    clear_structure_caches, structure_cache_info, FermionParityFusionRule, StructureCacheKind,
};

type FpSu2Rule = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

/// FNV-1a over 64-bit words: a bit-exact fingerprint of terms and outputs.
pub(super) fn fingerprint(words: impl IntoIterator<Item = u64>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// A platform-portable word for a pinned value: rounded to `f32`, with ±0
/// and every NaN folded. Why not the `f64` bits: symbol providers and dense
/// kernels may round the last bits differently across platforms (libm, FMA,
/// SIMD width; #2104), while a changed term set, order or accumulation
/// path moves values far beyond `f32` rounding. Same-run comparisons (cold
/// against warm) stay bit-exact.
pub(super) fn portable(value: f64) -> u64 {
    let value = value as f32;
    if value == 0.0 {
        0
    } else if value.is_nan() {
        u64::MAX
    } else {
        u64::from(value.to_bits())
    }
}

/// `(terms, f64 output, Complex64 output)` fingerprints of one MF trace:
/// term blocks exact, values through [`portable`].
type Prints = [u64; 3];

fn real(i: usize) -> f64 {
    ((i * 7919 + 13) % 1009) as f64 / 37.0 - 13.0
}

fn complex(i: usize) -> Complex64 {
    Complex64::new(real(i), real(i + 503) / 3.0)
}

fn mf_prints<R>(
    provider: Arc<R>,
    src_hom: FusionTreeHomSpace,
    axes: TensorTraceAxisSpec<'_>,
    dst_nout: usize,
) -> Prints
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + symtn_core::CheckedFusionAlgebra,
{
    let src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&provider),
        src_hom,
    )
    .unwrap();
    let dst_hom = crate::tensortrace_fusion_dyn_preflight_checked(&src, axes, dst_nout)
        .unwrap()
        .into_selected_homspace();
    let dst = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(provider, dst_hom)
        .unwrap();
    let structure =
        TensorTraceFusionStructure::<f64>::compile_fusion_dyn_checked(&dst, &src, axes).unwrap();
    let terms = fingerprint(structure.terms().iter().flat_map(|term| {
        [
            term.dst_block() as u64,
            term.src_block() as u64,
            portable(*term.coefficient()),
        ]
    }));
    let len = src.space().required_len().unwrap();
    let real_data = (0..len).map(real).collect::<Vec<_>>();
    let complex_data = (0..len).map(complex).collect::<Vec<_>>();
    let real_out =
        crate::tensortrace_fusion_dyn_owned_checked(&dst, &src, &real_data, axes, 1.0).unwrap();
    let complex_out = crate::tensortrace_fusion_dyn_owned_checked(
        &dst,
        &src,
        &complex_data,
        axes,
        Complex64::new(1.0, 0.0),
    )
    .unwrap();
    [
        terms,
        fingerprint(real_out.iter().map(|&value| portable(value))),
        fingerprint(
            complex_out
                .iter()
                .flat_map(|value| [portable(value.re), portable(value.im)]),
        ),
    ]
}

/// A named bit-pin case.
type Case<'a> = (&'a str, Prints, Box<dyn Fn() -> Prints + 'a>);

/// Runs a case cold (after a reset) and warm; both must equal `expected`.
fn assert_cold_and_warm(name: &str, expected: Prints, prints: impl Fn() -> Prints) {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    clear_structure_caches();
    let cold = prints();
    let warm = prints();
    assert_eq!(cold, warm, "{name}: warm differs from cold");
    assert_eq!(cold, expected, "{name}: differs from the pinned revision");
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

/// What: multiplicity-free trace terms and outputs keep the bits of
/// c7b82353/a04e59c2 (one `(p…, q…)` braid, the identity permutation, the
/// lazy adjoint, two pairs) for SU(2), the fermionic fZ2⊠SU(2) product
/// (twist factor) and U(1) (Unique), and a warm call equals a cold one.
#[test]
fn mf_trace_bits_match_pinned_revision_cold_and_warm() {
    let open = TensorTraceAxisSpec::new(&[1, 3], &[0], &[2]);
    let identity = TensorTraceAxisSpec::new(&[0, 2], &[1], &[3]);
    let adjoint = TensorTraceAxisSpec::new_with_conjugation(&[1, 3], &[0], &[2], true);
    let both = TensorTraceAxisSpec::new(&[], &[0, 1], &[2, 3]);
    let su2 = || Arc::new(SU2FusionRule);
    let fp = || Arc::new(FpSu2Rule::default());
    let u1 = || Arc::new(U1FusionRule);
    let cases: [Case; 10] = [
        (
            "su2 open",
            PIN_SU2_OPEN,
            Box::new(|| mf_prints(su2(), su2_hom(), open, 1)),
        ),
        (
            "su2 identity",
            PIN_SU2_IDENTITY,
            Box::new(|| mf_prints(su2(), su2_hom(), identity, 1)),
        ),
        (
            "su2 adjoint",
            PIN_SU2_ADJOINT,
            Box::new(|| mf_prints(su2(), su2_hom(), adjoint, 1)),
        ),
        (
            "su2 both",
            PIN_SU2_BOTH,
            Box::new(|| mf_prints(su2(), su2_hom(), both, 0)),
        ),
        (
            "fp⊠su2 open",
            PIN_FP_OPEN,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), open, 1)),
        ),
        (
            "fp⊠su2 identity",
            PIN_FP_IDENTITY,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), identity, 1)),
        ),
        (
            "fp⊠su2 adjoint",
            PIN_FP_ADJOINT,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), adjoint, 1)),
        ),
        (
            "fp⊠su2 both",
            PIN_FP_BOTH,
            Box::new(|| mf_prints(fp(), fp_su2_hom(), both, 0)),
        ),
        (
            "u1 open",
            PIN_U1_OPEN,
            Box::new(|| mf_prints(u1(), u1_hom(), open, 1)),
        ),
        (
            "u1 adjoint",
            PIN_U1_ADJOINT,
            Box::new(|| mf_prints(u1(), u1_hom(), adjoint, 1)),
        ),
    ];
    for (name, expected, prints) in cases {
        assert_cold_and_warm(name, expected, prints);
    }
}

/// Recorded through [`portable`] at 25724aa2, whose f64 bits equal
/// those of a04e59c2 (a revision without cache 4) on macOS arm64.
const PIN_SU2_OPEN: Prints = [
    11611427905873280266,
    10875861184201763681,
    18314849795358653513,
];
const PIN_SU2_IDENTITY: Prints = [
    4232214989127025742,
    14871361164243178883,
    7230379840223023516,
];
const PIN_SU2_ADJOINT: Prints = [
    11908699873957774648,
    10875861184201763681,
    595741395235360969,
];
const PIN_SU2_BOTH: Prints = [
    3784787345327458428,
    2335335494070420844,
    18406391343200945282,
];
const PIN_FP_OPEN: Prints = [
    6096770442742908238,
    9093196998694218356,
    13131061586690796996,
];
const PIN_FP_IDENTITY: Prints = [9754700174760375834, 954779586424958464, 90324878527814011];
const PIN_FP_ADJOINT: Prints = [
    7639453720453207797,
    18388007180403325428,
    8925810726760189892,
];
const PIN_FP_BOTH: Prints = [
    549713337742343836,
    13971276279694029957,
    4008062945554711502,
];
const PIN_U1_OPEN: Prints = [
    13709948616143683732,
    11264073318909681051,
    8471838890349698562,
];
const PIN_U1_ADJOINT: Prints = [
    10518078638784018340,
    13735893356685760447,
    15747945630213552074,
];

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
