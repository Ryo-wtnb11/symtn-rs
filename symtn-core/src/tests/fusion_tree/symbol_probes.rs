use super::*;

// --- Probe: complex-`Scalar` fusion rule through the recoupling
// engine. Most symtn-core providers use `Scalar = f64`; Fibonacci anyons
// need `Scalar = Complex64`. This probe rule
// is *not* a physical anyon model (its F-symbol is a constant 1, so it
// makes no pentagon claim) — it exists purely to prove that
// `multiplicity_free_braid_tree` / `FusionTermAccumulator` compile and
// run correctly when `Scalar: num_complex::Complex64` (Add/Mul/Clone from
// `num_complex`, plus a genuinely complex `CategoricalScalar::conj`).
#[derive(Clone, Copy, Debug)]
struct ComplexScalarProbeRule;

impl FusionRule for ComplexScalarProbeRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            // Fibonacci-shaped multi-channel fusion: x⊗x = {vacuum, x}.
            // This is what forces `FusionStyleKind::Simple` (not `Unique`)
            // and exercises the multi-term loop in the braid engine.
            (1, 1) => smallvec![SectorId::new(0), SectorId::new(1)],
            _ => SectorVec::new(),
        }
    }
}

impl MultiplicityFreeFusionRule for ComplexScalarProbeRule {}

const PROBE_ANGLE_ALPHA: f64 = std::f64::consts::FRAC_PI_3;

const PROBE_ANGLE_BETA: f64 = 2.0 * std::f64::consts::FRAC_PI_3;

impl MultiplicityFreeFusionSymbols for ComplexScalarProbeRule {
    type Scalar = num_complex::Complex64;

    // Trivial associator (1 on every allowed channel, since
    // `fusion_channels` already zeroes out disallowed ones via the
    // engine's `nsymbol` gate): this probe only needs to exercise the
    // complex-scalar plumbing, not satisfy the pentagon identity.
    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        num_complex::Complex64::new(1.0, 0.0)
    }

    // The one place a genuine complex phase enters: R^{xx}_vacuum = e^{iα},
    // R^{xx}_x = e^{iβ}, distinct angles so the two channels are
    // distinguishable in the assertions below.
    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        if self.nsymbol(left, right, coupled) == 0 {
            return num_complex::Complex64::new(0.0, 0.0);
        }
        if left.id() == 0 || right.id() == 0 {
            return num_complex::Complex64::new(1.0, 0.0);
        }
        if coupled.id() == 0 {
            num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_ALPHA)
        } else {
            num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_BETA)
        }
    }
}

#[test]
fn complex_scalar_r_symbol_and_conjugate_inverse_braid() {
    let rule = ComplexScalarProbeRule;
    for coupled in [0usize, 1usize] {
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1], coupled, [false, false], [], [1]).unwrap();
        let expected = num_complex::Complex64::from_polar(
            1.0,
            if coupled == 0 {
                PROBE_ANGLE_ALPHA
            } else {
                PROBE_ANGLE_BETA
            },
        );

        let forward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
        assert_eq!(forward.len(), 1);
        assert!((forward[0].1 - expected).norm() < 1.0e-12);

        // Reflected levels select the inverse-artin branch: the
        // coefficient must come back as the complex conjugate, proving
        // `CategoricalScalar::conj` (not just `Clone`/`Mul`) is wired through
        // for a non-real `Scalar`.
        let backward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[1, 0]).unwrap();
        assert_eq!(backward.len(), 1);
        assert!((backward[0].1 - expected.conj()).norm() < 1.0e-12);
    }
}

#[test]
fn complex_scalar_braid_tree_expands_multichannel_loop() {
    // Rank-3 tree with an index>0 swap: exercises the `fusion_channels(a,
    // d)` loop branch of `multiplicity_free_artin_braid_at_with_inverse`
    // (f_symbol_scalar * r_symbol_scalar * conj composition) —
    // this is the part of the engine Fibonacci's Simple-fusion braid
    // actually needs (the rank-2 probe above only reaches the
    // single-r-symbol `index == 0` branch).
    let rule = ComplexScalarProbeRule;
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1])
        .unwrap();

    let braided = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();

    // Hand-derived from the engine formula in the index>0 branch with
    // this rule's constant F=1: coefficient(c') = R(c,d,e) * conj(R(a,d,c')).
    // Here a=b=c=d=e=x, so R(c,d,e) = e^{iβ}; c'=vacuum -> R(a,d,c')=e^{iα},
    // c'=x -> e^{iβ}.
    assert_eq!(braided.len(), 2);
    let coeff_for = |innerline: usize| {
        braided
            .iter()
            .find(|(t, _)| t.innerlines() == [SectorId::new(innerline)])
            .unwrap()
            .1
    };
    let expected_vacuum_channel =
        num_complex::Complex64::from_polar(1.0, PROBE_ANGLE_BETA - PROBE_ANGLE_ALPHA);
    let expected_x_channel = num_complex::Complex64::new(1.0, 0.0);
    assert!((coeff_for(0) - expected_vacuum_channel).norm() < 1.0e-12);
    assert!((coeff_for(1) - expected_x_channel).norm() < 1.0e-12);
}

#[test]
fn fibonacci_low_rank_multi_fmoves_query_no_f_except_one_admissible_inverse_step() {
    let rule = FibonacciFAdmissibilityProbe::new();
    let vacuum = SectorId::new(0);
    let tau = SectorId::new(1);
    let empty = FusionTreeKey::new([], vacuum, [], [], []);
    let rank_one = FusionTreeKey::new([tau], tau, [false], [], []);
    let rank_two = FusionTreeKey::new(
        [tau, tau],
        vacuum,
        [false, false],
        [],
        [MultiplicityIndex::ONE],
    );

    // What: forward rank 0/1/2 and inverse rank 0/1 match the plain table
    // rule and do not enter an F-symbol provider.
    assert_eq!(
        SimpleK(&rule).multi_fmove(&empty),
        SimpleK(&TableRule::fibonacci()).multi_fmove(&empty)
    );
    assert_eq!(
        SimpleK(&rule).multi_fmove(&rank_one),
        SimpleK(&TableRule::fibonacci()).multi_fmove(&rank_one)
    );
    assert_eq!(
        SimpleK(&rule).multi_fmove(&rank_two),
        SimpleK(&TableRule::fibonacci()).multi_fmove(&rank_two)
    );
    assert_eq!(
        SimpleK(&rule).multi_fmove_inv(tau, tau, &empty, false),
        SimpleK(&TableRule::fibonacci()).multi_fmove_inv(tau, tau, &empty, false,)
    );
    assert_eq!(
        SimpleK(&rule).multi_fmove_inv(tau, vacuum, &rank_one, false),
        SimpleK(&TableRule::fibonacci()).multi_fmove_inv(tau, vacuum, &rank_one, false,)
    );
    assert!(rule.take_calls().is_empty());

    // What: inverse rank 2 still performs its one associator step, with
    // unchanged data and an admissible provider call.
    assert_eq!(
        SimpleK(&rule).multi_fmove_inv(tau, tau, &rank_two, false),
        SimpleK(&TableRule::fibonacci()).multi_fmove_inv(tau, tau, &rank_two, false,)
    );
    let calls = rule.take_calls();
    assert!(!calls.is_empty());
    assert_fibonacci_f_calls_are_admissible(&calls);
}

struct UniqueFAdmissibilityProbe {
    calls: std::sync::Mutex<Vec<[SectorId; 6]>>,
}

impl UniqueFAdmissibilityProbe {
    fn new() -> Self {
        Self {
            calls: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn take_calls(&self) -> Vec<[SectorId; 6]> {
        std::mem::take(
            &mut *self
                .calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }
}

impl FusionRule for UniqueFAdmissibilityProbe {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        Z2FusionRule.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        Z2FusionRule.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        Z2FusionRule.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        Z2FusionRule.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        Z2FusionRule.fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        Z2FusionRule.nsymbol(left, right, coupled)
    }
}

impl MultiplicityFreeFusionRule for UniqueFAdmissibilityProbe {}

impl MultiplicityFreeFusionSymbols for UniqueFAdmissibilityProbe {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push([left, middle, right, coupled, left_coupled, right_coupled]);
        let admissible = Z2FusionRule.nsymbol(left, middle, left_coupled) != 0
            && Z2FusionRule.nsymbol(left_coupled, right, coupled) != 0
            && Z2FusionRule.nsymbol(middle, right, right_coupled) != 0
            && Z2FusionRule.nsymbol(left, right_coupled, coupled) != 0;
        if admissible {
            Z2FusionRule.f_symbol_scalar(left, middle, right, coupled, left_coupled, right_coupled)
        } else {
            997.0
        }
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        Z2FusionRule.r_symbol_scalar(left, right, coupled)
    }
}

impl MultiplicityFreeRigidSymbols for UniqueFAdmissibilityProbe {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Z2FusionRule.dim_scalar(sector)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Z2FusionRule.inv_dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Z2FusionRule.sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Z2FusionRule.inv_sqrt_dim_scalar(sector)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        Z2FusionRule.twist_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        Z2FusionRule.frobenius_schur_phase_scalar(sector)
    }
}

#[test]
fn unique_multi_fmove_callers_preserve_admissible_z2_results() {
    let rule = UniqueFAdmissibilityProbe::new();
    let odd = z2_odd();
    let even = z2_even();
    let long = FusionTreeKey::new(
        [odd; 4],
        even,
        [false; 4],
        [even, odd],
        [MultiplicityIndex::ONE; 3],
    );
    let short = FusionTreeKey::new(
        [odd; 3],
        odd,
        [false; 3],
        [even],
        [MultiplicityIndex::ONE; 2],
    );

    // What: the Unique-fusion wrappers that share the associator boundary
    // retain exact output trees, coefficients, and conjugation direction.
    assert_eq!(
        multi_fmove_surgery(&UniqueK(&rule), &long),
        multi_fmove_surgery(&UniqueK(&Z2FusionRule), &long)
    );
    let calls = rule.take_calls();
    assert!(!calls.is_empty());
    for &[left, middle, right, coupled, left_coupled, right_coupled] in &calls {
        assert_ne!(Z2FusionRule.nsymbol(left, middle, left_coupled), 0);
        assert_ne!(Z2FusionRule.nsymbol(left_coupled, right, coupled), 0);
        assert_ne!(Z2FusionRule.nsymbol(middle, right, right_coupled), 0);
        assert_ne!(Z2FusionRule.nsymbol(left, right_coupled, coupled), 0);
    }

    assert_eq!(
        multi_fmove_inv_surgery(&UniqueK(&rule), &(odd, false), even, &short),
        multi_fmove_inv_surgery(&UniqueK(&Z2FusionRule), &(odd, false), even, &short)
    );
    let calls = rule.take_calls();
    assert!(!calls.is_empty());
    for &[left, middle, right, coupled, left_coupled, right_coupled] in &calls {
        assert_ne!(Z2FusionRule.nsymbol(left, middle, left_coupled), 0);
        assert_ne!(Z2FusionRule.nsymbol(left_coupled, right, coupled), 0);
        assert_ne!(Z2FusionRule.nsymbol(middle, right, right_coupled), 0);
        assert_ne!(Z2FusionRule.nsymbol(left, right_coupled, coupled), 0);
    }
}

#[test]
fn fibonacci_braid_tree_end_to_end_matches_hand_derived_coefficients() {
    // End-to-end: Simple fusion + Anyonic braiding + complex Scalar
    // through `multiplicity_free_braid_tree` on a rank-3 tree, with
    // coefficients hand-derived from the engine's own
    // R(c,d,e) * conj(F(d,a,b,e,c',c) * R(a,d,c')) formula
    // (`multiplicity_free_artin_braid_at_with_inverse`, index > 0
    // branch) substituting TensorKitSectors' F/R values directly.
    let rule = TableRule::fibonacci();
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1])
        .unwrap();

    let braided = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
    assert_eq!(braided.len(), 2);

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);
    let coeff_for = |innerline: usize| {
        braided
            .iter()
            .find(|(t, _)| t.innerlines() == [SectorId::new(innerline)])
            .unwrap()
            .1
    };

    let expected_vacuum_channel = Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(3.0 / 5.0);
    let expected_tau_channel = Complex64::new(-1.0 / phi, 0.0);
    assert!((coeff_for(0) - expected_vacuum_channel).norm() < 1.0e-12);
    assert!((coeff_for(1) - expected_tau_channel).norm() < 1.0e-12);
}

#[test]
fn fibonacci_elementary_artin_rows_match_tensorkit_coefficients() {
    // What: the private elementary Artin operation preserves TensorKit's
    // destination order and the independently substituted F/R coefficients
    // for both crossing orientations.
    let rule = TableRule::fibonacci();
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1])
        .unwrap();
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let cispi = |x: f64| Complex64::from_polar(1.0, std::f64::consts::PI * x);

    let forward = multiplicity_free_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap();
    let inverse = multiplicity_free_artin_braid_at_with_inverse(&rule, &tree, 1, true).unwrap();

    assert_eq!(
        forward
            .iter()
            .map(|(key, _)| key.innerlines()[0].id())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        inverse
            .iter()
            .map(|(key, _)| key.innerlines()[0].id())
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    let forward_expected = [
        Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(3.0 / 5.0),
        Complex64::new(-1.0 / phi, 0.0),
    ];
    let inverse_expected = [
        Complex64::new(1.0 / phi.sqrt(), 0.0) * cispi(-3.0 / 5.0),
        Complex64::new(-1.0 / phi, 0.0),
    ];
    for ((_, actual), expected) in forward.iter().zip(forward_expected) {
        assert!((*actual - expected).norm() < 1.0e-12);
    }
    for ((_, actual), expected) in inverse.iter().zip(inverse_expected) {
        assert!((*actual - expected).norm() < 1.0e-12);
    }
}
