//! Checked Generic tree-pair braid/permute at the symtn-core boundary.
//!
//! Value oracle: under a symmetric braiding, exchanging the two equal legs of
//! a splitting vertex `x ⊗ x → c` acts on its channel space as the exchange
//! operator, whose eigenvalue is +1 on Sym²(x) and −1 on Λ²(x). For SU(3),
//! Sym²(8) = 1 ⊕ 8 ⊕ 27, Λ²(8) = 8 ⊕ 10 ⊕ 10̄, Sym²(3) = 6, Λ²(3) = 3̄.
//! A multiplicity-free channel therefore gets the gauge-invariant
//! coefficient ±1, and the two-dimensional 8-channel of 8 ⊗ 8 gets an
//! orthogonal matrix with eigenvalues {+1, −1} (trace 0, determinant −1).
//! TensorKit `permute`/`braid` on a tree pair (bend all legs into the
//! codomain, braid, bend back) must reproduce these, on either side.

use super::*;
use std::cell::Cell;

const TOL: f64 = 1e-12;

fn sun3() -> LieGroupFusionRule {
    symtn_sectors::testing::su_lie_rule(3)
}

fn irrep(rule: &LieGroupFusionRule, dynkin: [i64; 2]) -> SectorId {
    rule.encode_irrep(&dynkin).unwrap()
}

fn vertex(label: usize) -> MultiplicityIndex {
    MultiplicityIndex::new(label).unwrap()
}

/// Codomain `[x, x] → c` (vertex `mu`), domain `[c] → c`.
fn codomain_pair(x: SectorId, c: SectorId, mu: usize) -> FusionTreePairKey {
    FusionTreePairKey::pair(
        FusionTreeKey::new([x, x], c, [false, false], [], [vertex(mu)]),
        FusionTreeKey::new([c], c, [false], [], []),
    )
}

/// Codomain `[c] → c`, domain `[x, x] → c` (vertex `mu`).
fn domain_pair(x: SectorId, c: SectorId, mu: usize) -> FusionTreePairKey {
    let pair = codomain_pair(x, c, mu);
    FusionTreePairKey::pair(pair.domain_tree().clone(), pair.codomain_tree().clone())
}

/// The two exchanges under test: the codomain legs of [`codomain_pair`] and
/// the domain legs of [`domain_pair`] (permutation indices: codomain, then
/// domain, as TensorKit's `permute(f₁, f₂, p₁, p₂)`).
fn exchanges(
    x: SectorId,
    c: SectorId,
    mu: usize,
) -> [(FusionTreePairKey, &'static [usize], &'static [usize]); 2] {
    [
        (codomain_pair(x, c, mu), &[1, 0], &[2]),
        (domain_pair(x, c, mu), &[0], &[2, 1]),
    ]
}

fn assert_single_term(
    terms: &[(FusionTreePairKey, f64)],
    expected_key: &FusionTreePairKey,
    expected: f64,
    label: &str,
) {
    assert_eq!(terms.len(), 1, "{label}: {terms:?}");
    assert_eq!(&terms[0].0, expected_key, "{label}");
    assert!(
        (terms[0].1 - expected).abs() < TOL,
        "{label}: {} != {expected}",
        terms[0].1
    );
}

#[test]
fn checked_generic_exchange_of_equal_legs_is_the_symmetry_sign_of_the_channel() {
    let rule = sun3();
    let three = irrep(&rule, [1, 0]);
    let eight = irrep(&rule, [1, 1]);
    let cases = [
        (eight, irrep(&rule, [0, 0]), 1.0, "8⊗8→1"),
        (eight, irrep(&rule, [2, 2]), 1.0, "8⊗8→27"),
        (eight, irrep(&rule, [3, 0]), -1.0, "8⊗8→10"),
        (eight, irrep(&rule, [0, 3]), -1.0, "8⊗8→10̄"),
        (three, irrep(&rule, [2, 0]), 1.0, "3⊗3→6"),
        (three, irrep(&rule, [0, 1]), -1.0, "3⊗3→3̄"),
    ];
    for (x, c, sign, label) in cases {
        for (pair, codomain, domain) in exchanges(x, c, 1) {
            let permuted = generic_permute_tree_pair_checked(&rule, &pair, codomain, domain)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_single_term(&permuted, &pair, sign, label);
            // A symmetric braiding ignores levels: the reversed level order
            // (an inverse braid under a non-symmetric rule) gives the same row.
            let n1 = codomain.len();
            let reversed = (0..codomain.len() + domain.len()).rev().collect::<Vec<_>>();
            let braided = generic_braid_tree_pair_checked(
                &rule,
                &pair,
                codomain,
                domain,
                &reversed[..n1],
                &reversed[n1..],
            )
            .unwrap();
            assert_single_term(&braided, &pair, sign, label);
        }
    }
}

#[test]
fn checked_generic_exchange_on_a_multiplicity_channel_has_eigenvalues_plus_and_minus_one() {
    let rule = sun3();
    let eight = irrep(&rule, [1, 1]);
    for side in 0..2 {
        // matrix[nu][mu]: coefficient of destination vertex nu from source mu.
        let mut matrix = [[0.0_f64; 2]; 2];
        for mu in 1..=2 {
            let (pair, codomain, domain) = exchanges(eight, eight, mu)[side].clone();
            for (destination, coefficient) in
                generic_permute_tree_pair_checked(&rule, &pair, codomain, domain).unwrap()
            {
                let nu = (0..2)
                    .find(|&nu| destination == exchanges(eight, eight, nu + 1)[side].0)
                    .unwrap_or_else(|| panic!("side {side}: unexpected {destination:?}"));
                matrix[nu][mu - 1] = coefficient;
            }
        }
        let [[a, b], [c, d]] = matrix;
        assert!((a + d).abs() < TOL, "side {side}: trace {matrix:?}");
        assert!(
            (a * d - b * c + 1.0).abs() < TOL,
            "side {side}: det {matrix:?}"
        );
        assert!((a * a + c * c - 1.0).abs() < TOL, "side {side}: {matrix:?}");
        assert!((a * b + c * d).abs() < TOL, "side {side}: {matrix:?}");
    }
}

#[test]
fn checked_su3_octet_duality_maps_match_tensorkit_literals() {
    // TensorKit 0.17.1 f87ca7f: bendright/foldright on
    // FusionTreeBlock{SU3Irrep}(((o,o),(o,)),((false,false),(false,))),
    // o = SU3Irrep(1,1), exports I₂. Repartition 2→1 is this one bendright.
    let rule = sun3();
    let eight = irrep(&rule, [1, 1]);
    let expected = [[1.0_f64, 0.0], [0.0, 1.0]];
    for mu in 1..=2 {
        let source = codomain_pair(eight, eight, mu);
        validate_generic_fusion_tree_pair_checked(&rule, &source).unwrap();
        for operation in 0..3 {
            let terms = match operation {
                0 => generic_bendright_tree_pair_checked(&rule, &source),
                1 => generic_foldright_tree_pair_result(&rule, &source),
                _ => generic_repartition_tree_pair_checked(&rule, &source, 1),
            }
            .unwrap();
            let mut actual = [0.0; 2];
            for (destination, coefficient) in terms {
                assert_eq!(destination.codomain_tree(), source.domain_tree());
                let domain = destination.domain_tree();
                assert_eq!(domain.uncoupled(), &[eight, eight]);
                assert_eq!(domain.coupled(), eight);
                assert_eq!(domain.is_dual(), &[operation == 1, operation != 1]);
                assert!(domain.innerlines().is_empty());
                assert_eq!(domain.vertices().len(), 1);
                actual[domain.vertices()[0].get() - 1] += coefficient;
            }
            for nu in 0..2 {
                assert!((actual[nu] - expected[mu - 1][nu]).abs() < TOL);
            }
            assert!((actual[mu - 1] - actual[2 - mu]).abs() > 0.5);
        }
    }
}

/// SU(3) behind a provider that counts and can fail `N` queries, with a
/// configurable braiding style.
struct SunProbe {
    inner: LieGroupFusionRule,
    braiding: BraidingStyleKind,
    n_calls: Cell<usize>,
    fail_n: Option<usize>,
    symbol_calls: Cell<usize>,
}

impl SunProbe {
    fn new() -> Self {
        Self {
            inner: sun3(),
            braiding: BraidingStyleKind::Bosonic,
            n_calls: Cell::new(0),
            fail_n: None,
            symbol_calls: Cell::new(0),
        }
    }

    fn symbol(&self) {
        self.symbol_calls.set(self.symbol_calls.get() + 1);
    }
}

impl symtn_sectors::testing::Provider for SunProbe {}

impl CheckedGenericFusion for SunProbe {
    type Error = RigidSpyError;
    fn rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(&self.inner)
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        self.braiding
    }
    fn vacuum(&self) -> SectorId {
        CheckedGenericFusion::vacuum(&self.inner)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        Ok(CheckedGenericFusion::try_dual(&self.inner, sector).unwrap())
    }
    fn try_fusion_channels(&self, a: SectorId, b: SectorId) -> Result<SectorVec, Self::Error> {
        Ok(CheckedGenericFusion::try_fusion_channels(&self.inner, a, b).unwrap())
    }
    fn try_fusion_channels_in_table(
        &self,
        a: SectorId,
        b: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Ok(CheckedGenericFusion::try_fusion_channels_in_table(&self.inner, a, b).unwrap())
    }
    fn try_nsymbol(&self, a: SectorId, b: SectorId, c: SectorId) -> Result<usize, Self::Error> {
        let call = self.n_calls.get() + 1;
        self.n_calls.set(call);
        if self.fail_n == Some(call) {
            return Err(RigidSpyError::N);
        }
        Ok(CheckedGenericFusion::try_nsymbol(&self.inner, a, b, c).unwrap())
    }
}

impl CheckedGenericRigidSymbols for SunProbe {
    type Scalar = f64;
    fn try_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol();
        Ok(CheckedGenericRigidSymbols::try_dim_scalar(&self.inner, sector).unwrap())
    }
    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol();
        Ok(CheckedGenericRigidSymbols::try_sqrt_dim_scalar(&self.inner, sector).unwrap())
    }
    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol();
        Ok(CheckedGenericRigidSymbols::try_inv_sqrt_dim_scalar(&self.inner, sector).unwrap())
    }
    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.symbol();
        Ok(
            CheckedGenericRigidSymbols::try_frobenius_schur_phase_scalar(&self.inner, sector)
                .unwrap(),
        )
    }
    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        self.symbol();
        Ok(
            CheckedGenericRigidSymbols::try_f_symbol_generic(&self.inner, a, b, c, d, e, f)
                .unwrap(),
        )
    }
    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        self.symbol();
        Ok(CheckedGenericRigidSymbols::try_r_symbol_generic(&self.inner, a, b, c).unwrap())
    }
}

#[test]
fn checked_generic_braid_and_permute_reject_operation_errors_before_admission() {
    let probe = SunProbe::new();
    let eight = irrep(&probe.inner, [1, 1]);
    let pair = codomain_pair(eight, eight, 1);

    let error =
        generic_braid_tree_pair_checked(&probe, &pair, &[1, 0], &[2], &[0], &[2]).unwrap_err();
    assert!(
        matches!(
            error,
            CheckedGenericSymbolError::Core(CoreError::DimensionMismatch {
                expected: 2,
                actual: 1
            })
        ),
        "{error:?}"
    );
    assert_eq!(probe.n_calls.get(), 0, "level count precedes admission");

    let anyonic = SunProbe {
        braiding: BraidingStyleKind::Anyonic,
        ..SunProbe::new()
    };
    let error = generic_permute_tree_pair_checked(&anyonic, &pair, &[1, 0], &[2]).unwrap_err();
    assert!(
        matches!(
            error,
            CheckedGenericSymbolError::Core(CoreError::UnsupportedBraidingStyle { .. })
        ),
        "{error:?}"
    );
    assert_eq!(
        anyonic.n_calls.get(),
        0,
        "braiding style precedes admission"
    );

    // An invalid tree is rejected by admission, before any recoupling symbol.
    let unknown_vertex = codomain_pair(eight, eight, 3);
    let error =
        generic_permute_tree_pair_checked(&probe, &unknown_vertex, &[1, 0], &[2]).unwrap_err();
    assert!(
        matches!(
            error,
            CheckedGenericSymbolError::Core(CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            })
        ),
        "{error:?}"
    );
    assert_eq!(probe.symbol_calls.get(), 0);

    let failing = SunProbe {
        fail_n: Some(1),
        ..SunProbe::new()
    };
    let error = generic_permute_tree_pair_checked(&failing, &pair, &[1, 0], &[2]).unwrap_err();
    assert_rigid_provider_error(error, RigidSpyError::N);
    assert_eq!(failing.symbol_calls.get(), 0);
}

#[test]
fn checked_generic_pair_validation_admits_both_trees_and_one_coupled_sector() {
    let rule = sun3();
    let eight = irrep(&rule, [1, 1]);
    let one = irrep(&rule, [0, 0]);
    validate_generic_fusion_tree_pair_checked(&rule, &codomain_pair(eight, eight, 2)).unwrap();
    validate_generic_fusion_tree_pair_checked(&rule, &domain_pair(eight, eight, 2)).unwrap();

    // N(8, 8, 8) = 2, so vertex 3 is invalid on either side.
    for pair in [codomain_pair(eight, eight, 3), domain_pair(eight, eight, 3)] {
        assert!(
            matches!(
                validate_generic_fusion_tree_pair_checked(&rule, &pair),
                Err(CheckedGenericStructureError::Core(
                    CoreError::MalformedFusionTree {
                        message: "fusion tree vertex label exceeds its fusion multiplicity",
                    }
                ))
            ),
            "{pair:?}"
        );
    }

    let mismatched = FusionTreePairKey::pair(
        FusionTreeKey::new([eight, eight], eight, [false, false], [], [vertex(1)]),
        FusionTreeKey::new([one], one, [false], [], []),
    );
    assert!(matches!(
        validate_generic_fusion_tree_pair_checked(&rule, &mismatched),
        Err(CheckedGenericStructureError::Core(
            CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
        ))
    ));

    let failing = SunProbe {
        fail_n: Some(1),
        ..SunProbe::new()
    };
    assert!(matches!(
        validate_generic_fusion_tree_pair_checked(&failing, &codomain_pair(eight, eight, 1)),
        Err(CheckedGenericStructureError::Provider(RigidSpyError::N))
    ));
}

fn sun_8x8_to_8x8_structure(rule: &LieGroupFusionRule) -> Arc<BlockStructure> {
    let eight = irrep(rule, [1, 1]);
    let leg = || SectorLeg::new([(eight, 1)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    )
    .coupled_subblock_structure_from_leg_degeneracies_generic_checked(rule)
    .unwrap()
}

fn block_pair(structure: &BlockStructure, index: usize) -> FusionTreePairKey {
    match structure.block(index).unwrap().key() {
        BlockKey::FusionTree(key) => key.clone(),
        other => panic!("not a fusion-tree block: {other:?}"),
    }
}

/// Every source column of `columns` equals the per-pair checked permutation
/// of the source read in that orientation (TensorKit `_trace_permute!`: one
/// permutation per fusion block, adjoint keys are `(f₂, f₁)`).
#[test]
fn checked_generic_block_indexed_permute_matches_per_pair_in_both_orientations() {
    let rule = sun3();
    let structure = sun_8x8_to_8x8_structure(&rule);
    let sources = (0..structure.block_count()).collect::<Vec<_>>();
    // Coupled 1, 8 (N = 2 on both sides), 10, 10̄, 27.
    assert_eq!(sources.len(), 8);
    let cases: [(FusionTreePairOrientation, &[usize], &[usize]); 2] = [
        (FusionTreePairOrientation::Direct, &[1, 0], &[2, 3]),
        (FusionTreePairOrientation::Adjoint, &[0, 1], &[3, 2]),
    ];
    for (orientation, codomain, domain) in cases {
        let columns = generic_permute_tree_pair_block_indexed_checked(
            &rule,
            &structure,
            &sources,
            orientation,
            codomain,
            domain,
        )
        .unwrap();
        assert_eq!(columns.source_count(), sources.len());
        for (column, &index) in sources.iter().enumerate() {
            let stored = block_pair(&structure, index);
            let source = match orientation {
                FusionTreePairOrientation::Direct => stored,
                FusionTreePairOrientation::Adjoint => FusionTreePairKey::pair(
                    stored.domain_tree().clone(),
                    stored.codomain_tree().clone(),
                ),
            };
            let mut expected = map_terms(
                generic_permute_tree_pair_checked(&rule, &source, codomain, domain).unwrap(),
            );
            for &(row, coefficient) in columns.column(column) {
                let destination = &columns.destinations()[row];
                let want = expected.remove(destination).unwrap_or(0.0);
                assert!(
                    (coefficient - want).abs() < TOL,
                    "{orientation:?} {source:?} -> {destination:?}: {coefficient} != {want}"
                );
            }
            assert!(
                expected.values().all(|value| value.abs() < TOL),
                "{orientation:?} {source:?}: missing {expected:?}"
            );
        }
    }
}

#[test]
fn checked_generic_block_indexed_permute_admits_sources_in_order() {
    let rule = sun3();
    let structure = sun_8x8_to_8x8_structure(&rule);
    let out_of_range = structure.block_count();
    let permute = |provider: &SunProbe, sources: &[usize]| {
        generic_permute_tree_pair_block_indexed_checked(
            provider,
            &structure,
            sources,
            FusionTreePairOrientation::Direct,
            &[1, 0],
            &[2, 3],
        )
        .unwrap_err()
    };

    // The first source's provider failure wins over a later bad index ...
    let failing = SunProbe {
        fail_n: Some(1),
        ..SunProbe::new()
    };
    assert_rigid_provider_error(permute(&failing, &[0, out_of_range]), RigidSpyError::N);
    assert_eq!(failing.symbol_calls.get(), 0);

    // ... an admissible first source is not recoupled before a later bad
    // index is seen ...
    let probe = SunProbe::new();
    let error = permute(&probe, &[0, out_of_range]);
    assert!(
        matches!(
            error,
            CheckedGenericSymbolError::Core(CoreError::BlockIndexOutOfBounds { index, count: 8 })
                if index == out_of_range
        ),
        "{error:?}"
    );
    assert!(probe.n_calls.get() > 0, "source 0 is admitted first");
    assert_eq!(probe.symbol_calls.get(), 0);

    // ... and an earlier bad index wins over a later source's failure, with
    // no recoupling symbol evaluated before every source is admitted.
    let failing = SunProbe {
        fail_n: Some(1),
        ..SunProbe::new()
    };
    let error = permute(&failing, &[out_of_range, 0]);
    assert!(
        matches!(
            error,
            CheckedGenericSymbolError::Core(CoreError::BlockIndexOutOfBounds { index, count: 8 })
                if index == out_of_range
        ),
        "{error:?}"
    );
    assert_eq!(failing.n_calls.get(), 0);
    assert_eq!(failing.symbol_calls.get(), 0);
}
