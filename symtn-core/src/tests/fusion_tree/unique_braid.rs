use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct UncertifiedCustomSymbolsRule;

impl symtn_sectors::testing::Provider for UncertifiedCustomSymbolsRule {}

impl FusionRule for UncertifiedCustomSymbolsRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        smallvec![SectorId::new(left.id() ^ right.id())]
    }
}

impl MultiplicityFreeFusionRule for UncertifiedCustomSymbolsRule {}

impl MultiplicityFreeFusionSymbols for UncertifiedCustomSymbolsRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        2.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        coupled: SectorId,
    ) -> Self::Scalar {
        if coupled.id() == 0 {
            3.0
        } else {
            5.0
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ComplexAsymmetricUniqueRule;

impl symtn_sectors::testing::Provider for ComplexAsymmetricUniqueRule {}

impl FusionRule for ComplexAsymmetricUniqueRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        smallvec![SectorId::new((left.id() + right.id()) % 4)]
    }
}

impl MultiplicityFreeFusionRule for ComplexAsymmetricUniqueRule {}

impl MultiplicityFreeFusionSymbols for ComplexAsymmetricUniqueRule {
    type Scalar = Complex64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        let angle = match (left.id(), right.id()) {
            (1, 2) => std::f64::consts::FRAC_PI_3,
            (2, 1) => std::f64::consts::FRAC_PI_6,
            _ => 0.0,
        };
        Complex64::from_polar(1.0, angle)
    }
}

#[test]
fn fusion_rule_exposes_unique_outputs_and_nsymbol_separately() {
    let z2 = Z2FusionRule;
    let su2 = SU2FusionRule;

    assert_eq!(z2.fusion_style(), FusionStyleKind::Unique);
    assert_eq!(
        z2.fusion_channels(SectorId::new(1), SectorId::new(1))
            .to_vec(),
        vec![SectorId::new(0)]
    );
    assert_eq!(
        z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
        1
    );
    assert_eq!(
        z2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(1)),
        0
    );

    assert_eq!(su2.fusion_style(), FusionStyleKind::Simple);
    assert_eq!(
        su2.fusion_channels(SectorId::new(1), SectorId::new(1))
            .to_vec(),
        vec![SectorId::new(0), SectorId::new(2)]
    );
    assert_eq!(
        su2.nsymbol(SectorId::new(1), SectorId::new(1), SectorId::new(2)),
        1
    );
}

#[test]
fn multiplicity_free_symbols_are_a_separate_scalar_api() {
    let z2 = Z2FusionRule;

    assert_eq!(
        <Z2FusionRule as MultiplicityFreeFusionSymbols>::Scalar::one(),
        1.0
    );
    assert_eq!(
        z2.f_symbol_scalar(
            SectorId::new(1),
            SectorId::new(1),
            SectorId::new(1),
            SectorId::new(1),
            SectorId::new(0),
            SectorId::new(0),
        ),
        1.0
    );
    assert_eq!(
        z2.r_symbol_scalar(SectorId::new(1), SectorId::new(1), SectorId::new(0)),
        1.0
    );
}

#[test]
fn unique_artin_braid_first_allows_unit_crossing_without_braiding() {
    let tree = FusionTreeKey::try_from_sector_ids([0, 1], 1, [false, true], [], [1]).unwrap();

    let terms = multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[1, 0], &[0, 1]).unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(0)]);
    assert_eq!(braided.is_dual(), &[true, false]);
    assert_eq!(braided.coupled(), SectorId::new(1));
    assert!(braided.innerlines().is_empty());
    assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE]);
}

#[test]
fn unique_artin_braid_first_rejects_nonunit_crossing_without_braiding() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, false], [], [1]).unwrap();

    let err = multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[1, 0], &[0, 1]).unwrap_err();

    assert_eq!(
        err,
        CoreError::UnsupportedSectorBraid {
            left: SectorId::new(1),
            right: SectorId::new(1),
            style: BraidingStyleKind::NoBraiding,
        }
    );
}

#[test]
fn merge_fusion_trees_treats_rank_zero_as_the_tensor_unit() {
    let unit = FusionTreeKey::try_from_sector_ids([], 0, [], [], []).unwrap();
    let tree = FusionTreeKey::try_from_sector_ids([1, 0], 1, [true, false], [], [1]).unwrap();

    for (lhs, rhs) in [(&unit, &tree), (&tree, &unit)] {
        let terms = merge_fusion_trees_multiplicity_free(&Z2FusionRule, lhs, rhs, SectorId::new(1))
            .unwrap();
        assert_eq!(terms, vec![(tree.clone(), 1.0)]);
    }
}

#[test]
fn merge_fusion_trees_pins_nontrivial_su2_f_coefficients() {
    // What: merging a spin-1 tree into three spin-1/2 leaves is a genuine
    // associator expansion, not a pointed-rule relabelling.
    let lhs = FusionTreeKey::try_from_sector_ids([1, 1], 2, [false, false], [], [1]).unwrap();
    let rhs = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false; 3], [0], [1, 1]).unwrap();

    let terms =
        merge_fusion_trees_multiplicity_free(&SU2FusionRule, &lhs, &rhs, SectorId::new(1)).unwrap();

    assert_eq!(terms.len(), 2);
    assert_eq!(
        terms
            .iter()
            .map(|(tree, _)| tree.innerlines().to_vec())
            .collect::<Vec<_>>(),
        [
            vec![SectorId::new(2), SectorId::new(1), SectorId::new(2)],
            vec![SectorId::new(2), SectorId::new(3), SectorId::new(2)],
        ]
    );
    assert!((terms[0].1 + 1.0 / 3.0_f64.sqrt()).abs() < 1.0e-14);
    assert!((terms[1].1 - (2.0 / 3.0_f64).sqrt()).abs() < 1.0e-14);
}

#[test]
fn unique_artin_braid_first_uses_r_symbol_for_first_crossing() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, true], [], [1]).unwrap();

    let terms =
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[1, 0], &[0, 1]).unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, -1.0);
    assert_eq!(braided.uncoupled(), &[SectorId::new(1), SectorId::new(1)]);
    assert_eq!(braided.is_dual(), &[true, false]);
    assert_eq!(braided.coupled(), SectorId::new(0));
    assert!(braided.innerlines().is_empty());
    assert_eq!(braided.vertices(), &[MultiplicityIndex::ONE]);
}

#[test]
fn unique_artin_braid_first_uses_first_innerline_for_rank_three() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1])
        .unwrap();

    let terms =
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[1, 0, 2], &[0, 1, 2])
            .unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, -1.0);
    assert_eq!(
        braided.uncoupled(),
        &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
    );
    assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

#[test]
fn unique_artin_braid_at_updates_innerline_for_later_unit_crossing() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 0, 1], 0, [false, false, true], [1], [1, 1])
        .unwrap();

    let terms = multiplicity_free_braid_tree(&PlanarZ2Rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(
        braided.uncoupled(),
        &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!(braided.is_dual(), &[false, true, false]);
    assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

#[test]
fn unique_artin_braid_at_uses_f_and_r_symbols_for_later_crossing() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, true, false], [0], [1, 1])
        .unwrap();

    let terms =
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[0, 2, 1], &[0, 1, 2])
            .unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, -1.0);
    assert_eq!(
        braided.uncoupled(),
        &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
    );
    assert_eq!(braided.is_dual(), &[false, false, true]);
    assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

// Literal transcription of TensorKit `permutation2swaps`
// (`src/auxiliary/auxiliary.jl:13`), kept 1-based as written there; the
// independent oracle for the prepared Artin schedule. Returns 0-based swaps.
fn tensorkit_permutation2swaps(permutation: &[usize]) -> Vec<usize> {
    let mut p = permutation.iter().map(|&axis| axis + 1).collect::<Vec<_>>();
    let n = p.len();
    let mut swaps = Vec::new();
    for k in 1..n {
        // append!(swaps, (p[k] - 1):-1:k)
        let mut s = p[k - 1] - 1;
        while s >= k {
            swaps.push(s);
            s -= 1;
        }
        for l in (k + 1)..=n {
            if p[l - 1] < p[k - 1] {
                p[l - 1] += 1;
            }
        }
        p[k - 1] = k;
    }
    swaps.into_iter().map(|s| s - 1).collect()
}

// TensorKit `braid` level rule (`braiding_manipulations.jl:238-243`): each
// swap is inverse when the left level is higher, then the levels swap.
fn tensorkit_artin_steps(permutation: &[usize], levels: &[usize]) -> Vec<(usize, bool)> {
    let mut levels = levels.to_vec();
    tensorkit_permutation2swaps(permutation)
        .into_iter()
        .map(|s| {
            let inverse = levels[s] > levels[s + 1];
            levels.swap(s, s + 1);
            (s, inverse)
        })
        .collect()
}

fn all_permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for p in all_permutations(n - 1) {
        for i in 0..=p.len() {
            let mut q = p.clone();
            q.insert(i, n - 1);
            out.push(q);
        }
    }
    out
}

fn all_level_tuples(n: usize) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for _ in 0..n {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                (0..n.max(1)).map(move |level| {
                    let mut next = prefix.clone();
                    next.push(level);
                    next
                })
            })
            .collect();
    }
    out
}

#[test]
fn tensorkit_permutation2swaps_transcription_matches_known_orders() {
    assert_eq!(tensorkit_permutation2swaps(&[2, 0, 1]), vec![1, 0]);
    assert_eq!(tensorkit_permutation2swaps(&[3, 0, 2, 1]), vec![2, 1, 0, 2]);
}

#[test]
fn prepared_artin_schedule_matches_tensorkit_for_every_permutation_and_level_tuple() {
    // What: the one prepared schedule every braid executes equals TensorKit's
    // swap order and inverse flags, exhaustively for rank <= 5 including
    // level ties.
    for rank in 0..=5 {
        let level_tuples = all_level_tuples(rank);
        for permutation in all_permutations(rank) {
            for levels in &level_tuples {
                let prepared = PreparedTreeBraid::new(&permutation, levels, rank)
                    .unwrap()
                    .artin_steps
                    .iter()
                    .map(|step| (step.index, step.inverse))
                    .collect::<Vec<_>>();
                assert_eq!(
                    prepared,
                    tensorkit_artin_steps(&permutation, levels),
                    "{permutation:?} {levels:?}"
                );
            }
        }
    }
}

#[test]
fn unique_braid_tree_replays_tensorkit_swap_order_and_level_updates() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1])
        .unwrap();

    let terms =
        multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &[2, 0, 1], &[0, 1, 2])
            .unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(
        braided.uncoupled(),
        &[SectorId::new(1), SectorId::new(1), SectorId::new(1)]
    );
    assert_eq!(braided.is_dual(), &[false, false, false]);
    assert_eq!(braided.coupled(), SectorId::new(1));
    assert_eq!(braided.innerlines(), &[SectorId::new(0)]);
    assert_eq!(
        braided.vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

#[test]
fn unique_braid_tree_uses_inverse_artin_branch_from_levels() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

    let forward_terms =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0, 1]).unwrap();
    assert_eq!(forward_terms.len(), 1);
    let (braided_forward, forward) = forward_terms.into_iter().next().unwrap();
    let inverse_terms =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[1, 0]).unwrap();
    assert_eq!(inverse_terms.len(), 1);
    let (braided_inverse, inverse) = inverse_terms.into_iter().next().unwrap();

    assert_eq!(forward, 5.0);
    assert_eq!(inverse, 7.0);
    assert_eq!(braided_forward, braided_inverse);
    assert_eq!(
        braided_forward.uncoupled(),
        &[SectorId::new(2), SectorId::new(1)]
    );
    assert_eq!(braided_forward.coupled(), SectorId::new(3));
}

#[test]
fn unique_braid_tree_reflected_levels_select_inverse_artin_branch() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
    let levels = [3, 8];
    let min_level = levels.iter().copied().min().unwrap();
    let max_level = levels.iter().copied().max().unwrap();
    let reflected_levels = levels
        .iter()
        .map(|&level| min_level + max_level - level)
        .collect::<Vec<_>>();

    let forward_terms =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &levels).unwrap();
    assert_eq!(forward_terms.len(), 1);
    let (forward_tree, forward_coeff) = forward_terms.into_iter().next().unwrap();
    let inverse_terms =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &reflected_levels)
            .unwrap();
    assert_eq!(inverse_terms.len(), 1);
    let (inverse_tree, inverse_coeff) = inverse_terms.into_iter().next().unwrap();

    assert_eq!(reflected_levels, vec![8, 3]);
    assert_eq!(forward_tree, inverse_tree);
    assert_eq!(forward_coeff, 5.0);
    assert_eq!(inverse_coeff, 7.0);
}

#[test]
fn symmetric_unique_direct_braid_matches_artin_replay_exactly() {
    // What: every rank-4 fZ2 permutation has the exact tree and sign of
    // TensorKit's adjacent-Artin semantics, including multiplication order.
    let tree = FusionTreeKey::try_from_sector_ids(
        [1, 1, 0, 1],
        1,
        [false, true, false, true],
        [0, 0],
        [1, 1, 1],
    )
    .unwrap();
    let levels = [0, 1, 2, 3];
    let mut permutation = [0usize, 1, 2, 3];
    loop {
        let direct_terms =
            multiplicity_free_braid_tree(&FermionParityFusionRule, &tree, &permutation, &levels)
                .unwrap();
        assert_eq!(direct_terms.len(), 1);
        let direct = direct_terms.into_iter().next().unwrap();

        let mut replay_tree = UnhashedFusionTree::from(tree.clone());
        let mut replay_coefficient = 1.0;
        let mut replay_levels = levels;
        for swap in tensorkit_permutation2swaps(&permutation) {
            let inverse = replay_levels[swap] > replay_levels[swap + 1];
            let coefficient = apply_unique_artin_braid_at_with_inverse(
                &FermionParityFusionRule,
                &mut replay_tree,
                swap,
                inverse,
            )
            .unwrap();
            replay_coefficient *= coefficient;
            replay_levels.swap(swap, swap + 1);
        }
        assert_eq!(direct, (replay_tree.freeze(), replay_coefficient));

        let Some(pivot) =
            (0..permutation.len() - 1).rfind(|&index| permutation[index] < permutation[index + 1])
        else {
            break;
        };
        let successor = (pivot + 1..permutation.len())
            .rfind(|&index| permutation[index] > permutation[pivot])
            .unwrap();
        permutation.swap(pivot, successor);
        permutation[pivot + 1..].reverse();
    }
}

/// Z3×Z3 with trivial F and the alternating bicharacter
/// `R(a, b) = ω^{a₁b₂ − a₂b₁}`, `ω = e^{2πi/3}`: a symmetric braiding
/// (`R(a,b)·R(b,a) = 1`, twist `R(a,a) = 1`) that truthfully certifies the
/// trivial associator gauge yet has `R(a,b) ≠ R(b,a)`. Sector id
/// `a₁ + 3·a₂`.
#[derive(Clone, Copy, Debug)]
pub(super) struct AlternatingZ3SquaredRule;

impl AlternatingZ3SquaredRule {
    fn components(sector: SectorId) -> (usize, usize) {
        assert!(sector.id() < 9, "Z3×Z3 fixture sector must be in 0..9");
        (sector.id() % 3, sector.id() / 3)
    }

    fn sector(first: usize, second: usize) -> SectorId {
        SectorId::new(first % 3 + 3 * (second % 3))
    }

    /// `a₁b₂ − a₂b₁ mod 3`, the exponent of `ω` in `R(a, b)`.
    fn exponent(left: SectorId, right: SectorId) -> usize {
        let (a1, a2) = Self::components(left);
        let (b1, b2) = Self::components(right);
        (a1 * b2 + 2 * a2 * b1) % 3
    }

    fn omega_power(exponent: usize) -> Complex64 {
        Complex64::from_polar(
            1.0,
            2.0 * std::f64::consts::PI * (exponent % 3) as f64 / 3.0,
        )
    }
}

impl symtn_sectors::testing::Provider for AlternatingZ3SquaredRule {}

impl FusionRule for AlternatingZ3SquaredRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (a1, a2) = Self::components(sector);
        Self::sector(3 - a1, 3 - a2)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (a1, a2) = Self::components(left);
        let (b1, b2) = Self::components(right);
        smallvec![Self::sector(a1 + b1, a2 + b2)]
    }
}

impl MultiplicityFreeFusionRule for AlternatingZ3SquaredRule {}

impl MultiplicityFreeFusionSymbols for AlternatingZ3SquaredRule {
    type Scalar = Complex64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, _coupled: SectorId) -> Self::Scalar {
        Self::omega_power(Self::exponent(left, right))
    }
}

fn alternating_z3_squared_tree(uncoupled: &[SectorId], is_dual: &[bool]) -> FusionTreeKey {
    let rule = AlternatingZ3SquaredRule;
    let mut running = uncoupled[0];
    let mut innerlines = Vec::new();
    for (offset, &next) in uncoupled[1..].iter().enumerate() {
        running = rule.fusion_channels(running, next)[0];
        if offset + 2 < uncoupled.len() {
            innerlines.push(running);
        }
    }
    FusionTreeKey::try_new_for_rule(
        &rule,
        uncoupled.iter().copied(),
        running,
        is_dual.iter().copied(),
        innerlines,
        std::iter::repeat_n(MultiplicityIndex::ONE, uncoupled.len() - 1),
    )
    .unwrap()
}

#[test]
fn alternating_bicharacter_r_symbol_is_not_symmetric() {
    // What: the fixture is a genuine witness: the trace sectors (1,2), (1,1)
    // have R(v,w) = ω² and R(w,v) = ω, while the braiding stays symmetric.
    let v = AlternatingZ3SquaredRule::sector(1, 2);
    let w = AlternatingZ3SquaredRule::sector(1, 1);
    let rule = AlternatingZ3SquaredRule;
    let c = rule.fusion_channels(v, w)[0];
    let omega = AlternatingZ3SquaredRule::omega_power(1);
    assert!((rule.r_symbol_scalar(v, w, c) - omega * omega).norm() < 1.0e-15);
    assert!((rule.r_symbol_scalar(w, v, c) - omega).norm() < 1.0e-15);
    assert!(
        (rule.r_symbol_scalar(v, w, c) * rule.r_symbol_scalar(w, v, c) - Complex64::new(1.0, 0.0))
            .norm()
            < 1.0e-15
    );
}

#[test]
fn certified_trivial_gauge_direct_braid_matches_artin_and_hand_oracle() {
    // What: for a certifying symmetric rule with non-symmetric R, the direct
    // shortcut equals the Artin composition (TensorKit's `braid` definition)
    // and the hand product of R(source-left, source-right) over every inverted
    // source pair, for every permutation of ranks 2..=4 and every level tuple
    // (both braid directions), including a vacuum leg.
    let rule = AlternatingZ3SquaredRule;
    let s = AlternatingZ3SquaredRule::sector;
    let pool = [s(1, 2), s(1, 1), s(2, 1), s(0, 0)];
    for rank in 2..=4 {
        let uncoupled = &pool[..rank];
        let is_dual = (0..rank).map(|axis| axis % 2 == 1).collect::<Vec<_>>();
        let tree = alternating_z3_squared_tree(uncoupled, &is_dual);
        let level_tuples = all_level_tuples(rank);
        for permutation in all_permutations(rank) {
            let mut exponent = 0;
            for first in 0..rank {
                for second in first + 1..rank {
                    let first_out = permutation.iter().position(|&a| a == first).unwrap();
                    let second_out = permutation.iter().position(|&a| a == second).unwrap();
                    if first_out > second_out {
                        exponent +=
                            AlternatingZ3SquaredRule::exponent(uncoupled[first], uncoupled[second]);
                    }
                }
            }
            let hand = AlternatingZ3SquaredRule::omega_power(exponent);
            let permuted = permutation
                .iter()
                .map(|&a| uncoupled[a])
                .collect::<Vec<_>>();
            let permuted_dual = permutation.iter().map(|&a| is_dual[a]).collect::<Vec<_>>();
            let hand_tree = alternating_z3_squared_tree(&permuted, &permuted_dual);

            for levels in &level_tuples {
                let terms =
                    multiplicity_free_braid_tree(&rule, &tree, &permutation, levels).unwrap();
                assert_eq!(terms.len(), 1);
                let (direct_tree, direct) = terms.into_iter().next().unwrap();

                let mut replay_tree = UnhashedFusionTree::from(tree.clone());
                let mut replay = Complex64::new(1.0, 0.0);
                for (index, inverse) in tensorkit_artin_steps(&permutation, levels) {
                    replay *= apply_unique_artin_braid_at_with_inverse(
                        &rule,
                        &mut replay_tree,
                        index,
                        inverse,
                    )
                    .unwrap();
                }

                let context = format!("{permutation:?} {levels:?}");
                assert_eq!(direct_tree, replay_tree.freeze(), "{context}");
                assert_eq!(direct_tree, hand_tree, "{context}");
                assert!(
                    (direct - replay).norm() < 1.0e-13,
                    "{context}: {direct} vs {replay}"
                );
                assert!(
                    (direct - hand).norm() < 1.0e-13,
                    "{context}: {direct} vs {hand}"
                );
            }

            let (permuted_tree, permuted_coefficient) =
                unique_permute_tree(&rule, &tree, &permutation).unwrap();
            assert_eq!(permuted_tree, hand_tree, "{permutation:?}");
            assert!(
                (permuted_coefficient - hand).norm() < 1.0e-13,
                "{permutation:?}"
            );
        }
    }
}

#[test]
fn table_rule_never_takes_the_direct_unique_braid() {
    // What: a pointed `TableRule` is `Unique` but certifies neither a
    // symmetric braiding nor a trivial associator gauge, so its braids run
    // the Artin steps (which read F and R) and never the direct path, which
    // assumes F = 1. The semion has F^{sss}_s = −1.
    let semion = TableRule::by_uuid("37fa4d16-97c6-11f1-9c13-c70bad3f9e94").unwrap();
    assert_eq!(semion.fusion_style(), FusionStyleKind::Unique);
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1])
        .unwrap();
    let swap = [1usize, 0, 2];
    let mut out = UnhashedFusionTree::default();
    assert_eq!(
        unique_direct_braid(
            &semion,
            &UnhashedFusionTree::from(tree.clone()),
            |position| swap[position],
            &mut out,
        ),
        Ok(None)
    );
    // Positive control: fZ2 (symmetric, trivial gauge) takes it.
    assert!(unique_direct_braid(
        &FermionParityFusionRule,
        &UnhashedFusionTree::from(tree),
        |position| swap[position],
        &mut out,
    )
    .unwrap()
    .is_some());
}
