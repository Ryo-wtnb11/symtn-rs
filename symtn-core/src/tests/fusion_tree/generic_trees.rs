use super::*;

#[test]
fn split_fusion_tree_matches_tensorkit_front_tail_convention() {
    let rule = SU2FusionRule;
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    let one = SU2Irrep::from_twice_spin(2).sector_id();
    let tree = FusionTreeKey::new(
        [half, half, one],
        one,
        [false, false, true],
        [SectorId::new(0)],
        [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
    );

    let (front, tail) = split_fusion_tree(&rule, &tree, 2).unwrap();

    assert_eq!(front.uncoupled(), &[half, half]);
    assert_eq!(front.coupled(), SectorId::new(0));
    assert_eq!(front.is_dual(), &[false, false]);
    assert_eq!(front.innerlines(), &[]);
    assert_eq!(front.vertices(), &[MultiplicityIndex::ONE]);
    assert_eq!(tail.uncoupled(), &[SectorId::new(0), one]);
    assert_eq!(tail.coupled(), one);
    assert_eq!(tail.is_dual(), &[false, true]);
    assert_eq!(tail.innerlines(), &[]);
    assert_eq!(tail.vertices(), &[MultiplicityIndex::ONE]);
}

#[test]
fn generic_split_preserves_multiplicity_style_and_vertex_labels() {
    // What: splitting a valid Generic tree returns two keys that remain
    // valid for the same rule and retain the selected multiplicity basis.
    let rule = ToyOmRule;
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);
    let source = FusionTreeKey::try_new_for_rule(
        &rule,
        [a, a, a],
        a,
        [false; 3],
        [c],
        [
            MultiplicityIndex::new(2).expect("test multiplicity label is one-based"),
            MultiplicityIndex::ONE,
        ],
    )
    .unwrap();

    let (front, tail) = split_fusion_tree(&rule, &source, 2).unwrap();

    front.validate_for_rule(&rule).unwrap();
    tail.validate_for_rule(&rule).unwrap();
    assert_eq!(front.vertices(), &[MultiplicityIndex::new(2).unwrap()]);
    assert_eq!(tail.vertices(), &[MultiplicityIndex::ONE]);
}

#[test]
fn generic_identity_braid_rejects_an_inadmissible_source() {
    // What: the Generic identity shortcut is behind the same categorical
    // boundary as multiplicity-free tree operations.
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);
    let invalid = FusionTreeKey::new([a; 3], c, [false; 3], [c], [MultiplicityIndex::ONE; 2]);

    assert_eq!(
        generic_braid_tree(&ToyOmRule, &invalid, &[0, 1, 2], &[0, 1, 2]).unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        }
    );
}

#[derive(Debug, Default)]
struct CrossIncompatibleGenericRule {
    f_calls: std::sync::atomic::AtomicUsize,
}

impl CrossIncompatibleGenericRule {
    const VACUUM: usize = 0;
    const A: usize = 1;
    const B: usize = 2;
    const C: usize = 3;
    const X: usize = 4;
    const E: usize = 5;
    const D: usize = 6;
    const F: usize = 7;
    const G: usize = 8;
    const Q: usize = 9;
}

impl FusionRule for CrossIncompatibleGenericRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(Self::VACUUM)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let channel = match (left.id(), right.id()) {
            (Self::VACUUM, sector) | (sector, Self::VACUUM) => Some(sector),
            (Self::A, Self::B) => Some(Self::E),
            (Self::E, Self::C) => Some(Self::D),
            (Self::D, Self::X) => Some(Self::Q),
            (Self::B, Self::C) => Some(Self::F),
            (Self::F, Self::X) => Some(Self::G),
            (Self::G, Self::X) => Some(Self::F),
            (Self::A, Self::G) => Some(Self::Q),
            (Self::A, Self::Q) => Some(Self::G),
            (Self::Q, Self::X) => Some(Self::D),
            (Self::D, Self::C) => Some(Self::E),
            _ => None,
        };
        channel
            .map(|sector| smallvec![SectorId::new(sector)])
            .unwrap_or_default()
    }
}

impl GenericFusionSymbols for CrossIncompatibleGenericRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
        _d: SectorId,
        _e: SectorId,
        _f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        self.f_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        GenericFArray::new(vec![997.0], (1, 1, 1, 1))
    }

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let rows = self.nsymbol(a, b, c);
        let cols = self.nsymbol(b, a, c);
        GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
    }
}

#[test]
fn generic_multi_fmove_filters_cross_incompatible_trees_before_f() {
    // What: forward and inverse multi-F moves discard individually valid
    // long/short trees whose cross-tree F vertex does not exist, even when
    // the provider returns a nonzero sentinel outside its valid domain.
    let rule = CrossIncompatibleGenericRule::default();
    let sector = SectorId::new;
    let long = FusionTreeKey::new(
        [
            sector(CrossIncompatibleGenericRule::A),
            sector(CrossIncompatibleGenericRule::B),
            sector(CrossIncompatibleGenericRule::C),
            sector(CrossIncompatibleGenericRule::X),
        ],
        sector(CrossIncompatibleGenericRule::Q),
        [false; 4],
        [
            sector(CrossIncompatibleGenericRule::E),
            sector(CrossIncompatibleGenericRule::D),
        ],
        [MultiplicityIndex::ONE; 3],
    );
    let short = FusionTreeKey::new(
        [
            sector(CrossIncompatibleGenericRule::B),
            sector(CrossIncompatibleGenericRule::C),
            sector(CrossIncompatibleGenericRule::X),
        ],
        sector(CrossIncompatibleGenericRule::G),
        [false; 3],
        [sector(CrossIncompatibleGenericRule::F)],
        [MultiplicityIndex::ONE; 2],
    );
    long.validate_for_rule(&rule).unwrap();
    short.validate_for_rule(&rule).unwrap();

    assert!(generic_multi_fmove_tree(&rule, &long).unwrap().is_empty());
    assert!(generic_multi_fmove_inv_tree(
        &rule,
        sector(CrossIncompatibleGenericRule::A),
        sector(CrossIncompatibleGenericRule::Q),
        &short,
        false,
    )
    .unwrap()
    .is_empty());
    assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

fn generic_tree_pair(
    vertices_first: usize,
    vertices_second: usize,
) -> (FusionTreeKey, FusionTreeKey) {
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);
    let first = FusionTreeKey::new(
        [a, a],
        c,
        [false, false],
        [],
        [MultiplicityIndex::new(vertices_first).expect("test multiplicity label is one-based")],
    );
    let second = FusionTreeKey::new(
        [a, a],
        c,
        [false, false],
        [],
        [MultiplicityIndex::new(vertices_second).expect("test multiplicity label is one-based")],
    );
    (first, second)
}

#[test]
fn fusion_tree_key_generic_distinguishes_vertices() {
    // What: multiplicity vertices are always part of categorical identity;
    // fusion style comes from the provider rather than a duplicated flag.
    let (first, second) = generic_tree_pair(1, 2);
    assert_ne!(first, second);
    assert_ne!(first.cmp(&second), std::cmp::Ordering::Equal);

    let mut set = std::collections::HashSet::new();
    set.insert(first);
    set.insert(second);
    assert_eq!(
        set.len(),
        2,
        "Generic keys differing in vertices must stay distinct"
    );
}

#[test]
fn frozen_generic_enumeration_shares_one_arena() {
    // What: Generic fanout puts every tree into one arena, which a holder
    // charges once; an equal-content key built on its own has its own.
    let rule = ToyOmRule;
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);
    let trees =
        collect_generic_fusion_trees_for_coupled(&rule, &[a, a], &[false, false], &[a, a], c);
    assert!(trees.len() > 1);
    assert!(trees
        .windows(2)
        .all(|trees| trees[0].shares_arena(&trees[1])));

    let independent = FusionTreeKey::new(
        trees[0].uncoupled().iter().copied(),
        trees[0].coupled(),
        trees[0].is_dual().iter().copied(),
        trees[0].innerlines().iter().copied(),
        trees[0].vertices().iter().copied(),
    );
    assert_eq!(independent, trees[0]);
    assert!(!independent.shares_arena(&trees[0]));
    assert!(independent.arena_retained_bytes() > 0);
}

#[test]
fn checked_generic_artin_calls_only_required_symbols_and_preserves_failures() {
    let outer = ArtinSpy::new();
    generic_artin_braid_at_with_inverse_checked(&outer, &unitary_rank2_tree(1), 0, false).unwrap();
    assert_eq!((outer.f_calls.get(), outer.r_calls.get()), (0, 1));
    assert_eq!(outer.rigid_calls.get(), 0);
    let inner = ArtinSpy::new();
    generic_artin_braid_at_with_inverse_checked(&inner, &unitary_rank3_tree(1), 1, false).unwrap();
    assert!(inner.f_calls.get() > 0 && inner.r_calls.get() > 0);
    let fail_r = ArtinSpy {
        fail_r: Some(1),
        ..ArtinSpy::new()
    };
    assert!(matches!(
        generic_artin_braid_at_with_inverse_checked(&fail_r, &unitary_rank2_tree(1), 0, false),
        Err(CheckedGenericSymbolError::Provider(ArtinSpyError::R))
    ));
    let fail_f = ArtinSpy {
        fail_f: Some(1),
        ..ArtinSpy::new()
    };
    assert!(matches!(
        generic_artin_braid_at_with_inverse_checked(&fail_f, &unitary_rank3_tree(1), 1, false),
        Err(CheckedGenericSymbolError::Provider(ArtinSpyError::F))
    ));
}

#[test]
fn checked_generic_artin_rejects_categorical_symbol_shapes_before_indexing() {
    let bad_r = ArtinSpy {
        bad_r: true,
        ..ArtinSpy::new()
    };
    assert!(matches!(
        generic_artin_braid_at_with_inverse_checked(&bad_r, &unitary_rank2_tree(1), 0, false),
        Err(CheckedGenericSymbolError::Shape { symbol: "R", .. })
    ));
    let bad_f = ArtinSpy {
        bad_f: true,
        ..ArtinSpy::new()
    };
    assert!(matches!(
        generic_artin_braid_at_with_inverse_checked(&bad_f, &unitary_rank3_tree(1), 1, false),
        Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
    ));
}

#[test]
fn checked_generic_bend_and_repartition_match_unchecked_rows_and_order() {
    let pair = a4_dual_pair_rank2(2);
    let unchecked_bend = generic_bendright_tree_pair(&A4BendRule, &pair).unwrap();
    let checked = A4BendRule;
    let checked_bend = generic_bendright_tree_pair_checked(&checked, &pair).unwrap();
    assert_eq!(checked_bend, unchecked_bend);

    let unchecked_repartition = generic_repartition_tree_pair(&A4BendRule, &pair, 0).unwrap();
    let checked = A4BendRule;
    let checked_repartition = generic_repartition_tree_pair_checked(&checked, &pair, 0).unwrap();
    assert_eq!(checked_repartition, unchecked_repartition);

    let checked = A4BendRule;
    assert_eq!(
        generic_bendleft_tree_pair_checked(&checked, &unchecked_bend[0].0).unwrap(),
        generic_bendleft_tree_pair(&A4BendRule, &unchecked_bend[0].0).unwrap()
    );
}

#[test]
fn checked_generic_b_and_a_reject_malformed_categorical_f_shapes() {
    let bad_b = CheckedA4Spy {
        bad_b_f: true,
        ..CheckedA4Spy::new()
    };
    assert!(matches!(
        generic_bendright_tree_pair_checked(&bad_b, &a4_pair_rank2(1)),
        Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
    ));

    let bad_a = CheckedA4Spy {
        bad_a_f: true,
        ..CheckedA4Spy::new()
    };
    let t = a4_three();
    assert!(matches!(
        GenericRigidAccess::try_a_symbol_generic(&bad_a, t, t, t),
        Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
    ));
}

#[test]
fn refute_bendright_uses_b_row_not_column() {
    // End-to-end: bend the codomain vertex μ; the ν output distribution must
    // equal ROW μ of B (coeff = coeff0·Bmat[μ,ν], coeff0=1 here). A μ↔ν swap
    // in the Bmat.get(μ,ν) call would emit COLUMN μ instead.
    let rule = TransposeProbeRule;
    let s = SectorId::new(1);
    let b = tp_expected_b();
    for mu in 1..=2usize {
        // cod [1,1]->1 vertex μ ; dom [1]->1.
        let cod = FusionTreeKey::new(
            [s, s],
            s,
            [false, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([s], s, [false], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
        // What: the ν fanout shares one arena and differs only in its
        // appended vertex label.
        assert_eq!(out.len(), 2);
        for terms in out.windows(2) {
            let left = terms[0].0.domain_tree();
            let right = terms[1].0.domain_tree();
            assert!(Arc::ptr_eq(&left.arena, &right.arena));
            assert_eq!(left.uncoupled(), right.uncoupled());
            assert_eq!(left.is_dual(), right.is_dual());
            assert_eq!(left.innerlines(), right.innerlines());
            assert_ne!(left.vertices(), right.vertices());
        }
        // Collect coeff keyed by output domain vertex label (=ν+1).
        let mut got = [0.0f64; 2];
        for (key, coeff) in &out {
            let nu = key.domain_tree().vertices()[0].get(); // 1-based ν label
            got[nu - 1] = *coeff;
        }
        let row = &b[mu - 1];
        for nu in 0..2 {
            assert!(
                (got[nu] - row[nu]).abs() < 1e-12,
                "μ={mu}: ν={nu} coeff {} want ROW-μ {} (transpose ⇒ column-μ)",
                got[nu],
                row[nu]
            );
        }
        // Guard: distinguishable from the column (transposed) reading.
        let col = [b[0][mu - 1], b[1][mu - 1]];
        assert!(
            (got[0] - col[0]).abs() > 1e-9 || (got[1] - col[1]).abs() > 1e-9,
            "μ={mu}: row and column coincide — test cannot discriminate"
        );
    }
}

// ==== REFUTE (adversarial): coeff2 adjoint conj on GENUINELY complex data ====
//
// Gap found by the verifier: in `ComplexUnitaryRule` the domain vector
// `coeff2` is always a real UNIT vector (rank-1 domain → seed case), so the
// `coeff₂'` adjoint (TK `duality_manipulations.jl:279`) and the
// `multi_Fmove_inv = conj(associator)` step (TK `basic_manipulations.jl:
// 439/462`) are NEVER exercised on complex data by any existing test — the
// A4 oracle is fully real, and the complex fold round-trip cancels a
// consistent double conj error.
//
// This synthetic (deliberately NOT pentagon-consistent — a pure algebraic
// fixture) rule drives one COMPLEX interior F into `coeff2` while keeping the
// A-matrix REAL, isolating the two conj sites:
//   * F(1,1,2,2,0,3) = 1  (real)  ⇒ Asymbol(1,2,3) is real, = 1.
//   * F(1,3,3,2,2,3) = w  (complex) ⇒ multi_associator seed = w.
// TK's `multi_Fmove_inv` returns conj(associator) = conj(w); TK's foldright
// contracts coeff₂' (a SECOND conj) against transpose(A)·coeff₁, so the two
// conjs cancel and the observable foldright coefficient is the RAW
// associator w. Test A pins `multi_Fmove_inv = conj(w)` alone (breaks the
// double-error symmetry); Test B pins the foldright net = w. Together they
// rule out both a single and a double conj slip. A single missing conj at
// EITHER site flips the observable to conj(w) ≠ w.
#[derive(Clone, Copy, Debug)]
struct Coeff2ConjRule;

fn c2c_w() -> Complex64 {
    Complex64::new(0.6, 0.8) // |w| = 1, genuinely complex
}

impl FusionRule for Coeff2ConjRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        sector // all self-dual
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(0)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(3)],
            (1, 3) | (3, 1) => smallvec![SectorId::new(2)],
            (2, 3) | (3, 2) => smallvec![SectorId::new(2)],
            (3, 3) => smallvec![SectorId::new(3)],
            _ => smallvec![SectorId::new(0)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        usize::from(self.fusion_channels(left, right).contains(&coupled))
    }
}

impl GenericFusionSymbols for Coeff2ConjRule {
    type Scalar = Complex64;
    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
        match ids {
            // Asymbol(1,2,3) reads F(dual1,1,2,2,0,3) = F(1,1,2,2,0,3): REAL.
            (1, 1, 2, 2, 0, 3) => GenericFArray::new(vec![Complex64::new(1.0, 0.0)], (1, 1, 1, 1)),
            // multi_associator seed for domain [3,3]->3 folded onto b=2: COMPLEX.
            (1, 3, 3, 2, 2, 3) => GenericFArray::new(vec![c2c_w()], (1, 1, 1, 1)),
            _ => {
                let shape = (
                    self.nsymbol(a, b, e),
                    self.nsymbol(e, c, d),
                    self.nsymbol(b, c, f),
                    self.nsymbol(a, f, d),
                );
                if shape == (1, 1, 1, 1) {
                    GenericFArray::new(vec![Complex64::new(1.0, 0.0)], shape)
                } else {
                    panic!("Coeff2ConjRule: unmodelled non-singleton F{ids:?} shape={shape:?}");
                }
            }
        }
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericRMatrix::new(vec![Complex64::new(1.0, 0.0)], 1, 1)
    }
}

impl GenericRigidSymbols for Coeff2ConjRule {
    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }
}

// Test A: `multi_Fmove_inv` alone returns conj(associator) on complex F.
#[test]
fn refute_multi_fmove_inv_is_conj_associator_complex() {
    let rule = Coeff2ConjRule;
    let s1 = SectorId::new(1);
    let s3 = SectorId::new(3);
    let w = c2c_w();
    // domain tree [3,3] -> 3 (single vertex); leading dual(a)=1, target b=2.
    let domain = FusionTreeKey::new([s3, s3], s3, [false, false], [], [MultiplicityIndex::ONE]);
    let terms = generic_multi_fmove_inv_tree(&rule, s1, SectorId::new(2), &domain, true).unwrap();
    assert_eq!(terms.len(), 1, "expected a single recoupled candidate");
    let (_, coeff) = &terms[0];
    assert_eq!(coeff.len(), 1, "coeff2 must be length-1 here");
    // Independent TK reading: inv coeff = conj(seed associator) = conj(w).
    assert!(
        (coeff[0] - w.conj()).norm() < 1e-12,
        "multi_Fmove_inv gave {} want conj(w)={} (A2 conj missing?)",
        coeff[0],
        w.conj()
    );
    // Discriminating: conj(w) must differ from w so the check has teeth.
    assert!((w - w.conj()).norm() > 0.1, "w not complex enough");
}

// Test B: foldright net observable = raw associator w (the two conjs cancel).
// A single dropped conj at EITHER site would surface as conj(w).
#[test]
fn refute_foldright_net_is_raw_associator_complex() {
    let rule = Coeff2ConjRule;
    let s1 = SectorId::new(1);
    let s2 = SectorId::new(2);
    let s3 = SectorId::new(3);
    let w = c2c_w();
    // codomain [1,2] -> 3 (coeff1 = unit, A = Asymbol(1,2,3) real = 1),
    // domain [3,3] -> 3 (drives complex coeff2).
    let codomain = FusionTreeKey::new([s1, s2], s3, [false, false], [], [MultiplicityIndex::ONE]);
    let domain = FusionTreeKey::new([s3, s3], s3, [false, false], [], [MultiplicityIndex::ONE]);
    let pair = FusionTreePairKey::pair(codomain, domain);
    let out = generic_foldright_tree_pair(&rule, &pair).unwrap();
    assert_eq!(out.len(), 1, "expected a single folded term");
    let coeff = out[0].1;
    assert!(
        (coeff - w).norm() < 1e-12,
        "foldright net = {coeff} want raw associator w={w} (odd # of conj slips ⇒ conj(w))"
    );
    // The wrong (single-conj-dropped) answer is conj(w); prove distinguishable.
    assert!(
        (coeff - w.conj()).norm() > 0.1,
        "test cannot discriminate conj(w) from w"
    );
}

#[test]
fn generic_full_key_block_composition_matches_per_source_replay() {
    let rule = Su3BendRule;
    let s42 = SectorId::new(1);
    let s31 = SectorId::new(2);
    let basis = (1..=2)
        .map(|mu| {
            FusionTreePairKey::pair(
                FusionTreeKey::new(
                    [s42, s31],
                    s31,
                    [false, false],
                    [],
                    [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
                ),
                FusionTreeKey::new([s31], s31, [false], [], []),
            )
        })
        .collect::<Vec<_>>();
    let mut columns = DenseColumns::with_capacity(basis.len(), basis.len());
    for source in 0..basis.len() {
        let row = columns.push_empty_row();
        columns.row_mut(row)[source] = Some(1.0);
    }

    let (dst_basis, dst_columns) = compose_block_terms(&basis, &columns, |key| {
        generic_bendright_tree_pair(&rule, key)
    })
    .unwrap();

    let oracle = basis
        .iter()
        .map(|source| {
            compose_terms(vec![(source.clone(), 1.0)], |key| {
                generic_bendright_tree_pair(&rule, key)
            })
            .unwrap()
        })
        .collect::<Vec<_>>();
    let mut expected_order = Vec::<FusionTreePairKey>::new();
    for rows in &oracle {
        for (key, _) in rows {
            if !expected_order.iter().any(|existing| existing == key) {
                expected_order.push(key.clone());
            }
        }
    }
    assert_eq!(dst_basis, expected_order);
    assert_eq!(dst_columns.num_src, basis.len());
    assert_eq!(dst_columns.num_rows, dst_basis.len());
    assert_eq!(oracle.len(), basis.len());
    for (destination_row, destination) in dst_basis.iter().enumerate() {
        let domain_vertices = destination.domain_tree().vertices();
        assert_eq!(
            domain_vertices.last().map(|index| index.get()),
            Some(destination_row + 1)
        );
        for (source, oracle_row) in oracle.iter().enumerate() {
            let got = dst_columns.row(destination_row)[source].unwrap_or(0.0);
            let want = oracle_row
                .iter()
                .find_map(|(key, coeff)| (key == destination).then_some(*coeff))
                .unwrap_or(0.0);
            assert!((got - want).abs() < 1e-12);
        }
    }
}

#[test]
fn checked_generic_cyclic_transpose_matches_unchecked_rows() {
    let rule = A4FoldRule;
    let t = SectorId::new(3);
    let checked = InfallibleGeneric::new(&rule);
    for (pair, codomain, domain, label) in [
        (
            FusionTreePairKey::pair(
                a4f_rank3(3, 2, 1),
                FusionTreeKey::new([t], t, [false], [], []),
            ),
            vec![1, 2, 3],
            vec![0],
            "clockwise",
        ),
        (
            FusionTreePairKey::pair(
                a4f_rank3(3, 1, 2),
                FusionTreeKey::new([t], t, [false], [], []),
            ),
            vec![3, 0, 1],
            vec![2],
            "anticlockwise",
        ),
        (
            FusionTreePairKey::pair(
                FusionTreeKey::new(
                    [t, t, t],
                    t,
                    [true, false, true],
                    [t],
                    [MultiplicityIndex::new(2).unwrap(), MultiplicityIndex::ONE],
                ),
                FusionTreeKey::new([t], t, [true], [], []),
            ),
            vec![2, 3, 0],
            vec![1],
            "multi-step dual flags",
        ),
    ] {
        let unchecked =
            map_terms(generic_transpose_tree_pair(&rule, &pair, &codomain, &domain).unwrap());
        let actual = map_terms(
            generic_transpose_tree_pair_checked(&checked, &pair, &codomain, &domain).unwrap(),
        );
        assert_term_maps_eq(&actual, &unchecked, label);
    }
}

#[test]
fn checked_generic_cyclic_transpose_preserves_provider_and_shape_errors() {
    let pair = a4_pair_rank2(1);
    let provider_error = CheckedA4Spy {
        fail_f: Some(1),
        ..CheckedA4Spy::new()
    };
    assert_rigid_provider_error(
        generic_transpose_tree_pair_checked(&provider_error, &pair, &[1, 2], &[0]).unwrap_err(),
        RigidSpyError::F,
    );

    let malformed = CheckedA4Spy {
        bad_a_f: true,
        ..CheckedA4Spy::new()
    };
    assert!(matches!(
        generic_transpose_tree_pair_checked(&malformed, &pair, &[1, 2], &[0]),
        Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
    ));
}

// What: the Generic ordered block (keyed driver, `braid_codomain`, and the
// shared ordered-map builder) equals the per-pair Generic transform for every
// source, and the fixtures reach both ordered storages: one destination per
// source (singleton columns) and fan-out (dense destination × source).
#[test]
fn generic_ordered_blocks_match_per_pair_in_both_storages() {
    use std::collections::BTreeMap;
    // HomSpace enumeration below builds coupled grids, which tests that count
    // those builds observe globally.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = A4FoldRule;
    let sectors = [SectorId::new(3)];
    let leg = |dual: bool| SectorLeg::new(sectors.iter().map(|&s| (s, 1)), dual);
    let (mut singleton, mut dense, mut braided) = (0, 0, 0);
    for (nc, nd) in [(2usize, 1usize), (1, 2), (2, 2)] {
        let n = nc + nd;
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new((0..nc).map(|i| leg(i % 2 == 1)).collect::<Vec<_>>()),
            FusionProductSpace::new((0..nd).map(|i| leg(i % 2 == 0)).collect::<Vec<_>>()),
        );
        let mut blocks: BTreeMap<String, Vec<FusionTreePairKey>> = BTreeMap::new();
        for key in hom.fusion_tree_keys_generic(&rule).unwrap() {
            blocks
                .entry(format!(
                    "{:?}{:?}{:?}{:?}",
                    key.codomain_tree().uncoupled(),
                    key.domain_tree().uncoupled(),
                    key.codomain_tree().is_dual(),
                    key.domain_tree().is_dual()
                ))
                .or_default()
                .push(key);
        }
        let reversed = (0..n).rev().collect::<Vec<_>>();
        let rotated = (1..n).chain([0]).collect::<Vec<_>>();
        let (codomain_levels, domain_levels) = reversed.split_at(nc);
        for sources in blocks.values() {
            for permutation in [&reversed, &rotated] {
                for split in 0..=n {
                    let (cp, dp) = permutation.split_at(split);
                    let cases = [
                        (
                            generic_braid_tree_pair_block_ordered(
                                &rule,
                                sources,
                                cp,
                                dp,
                                codomain_levels,
                                domain_levels,
                            ),
                            sources
                                .iter()
                                .map(|source| {
                                    generic_braid_tree_pair(
                                        &rule,
                                        source,
                                        cp,
                                        dp,
                                        codomain_levels,
                                        domain_levels,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        ),
                        (
                            generic_transpose_tree_pair_block_ordered(&rule, sources, cp, dp),
                            sources
                                .iter()
                                .map(|source| generic_transpose_tree_pair(&rule, source, cp, dp))
                                .collect::<Vec<_>>(),
                        ),
                    ];
                    for (case, (block, per_pair)) in cases.into_iter().enumerate() {
                        let Ok(block) = block else {
                            assert!(per_pair.iter().any(Result::is_err));
                            continue;
                        };
                        braided += usize::from(case == 0);
                        let coefficient = |destination: usize, source: usize| match block.storage()
                        {
                            OrderedBlockLinearStorage::SingletonColumns {
                                destination_rows,
                                coefficients,
                            } => (destination_rows[source] == destination)
                                .then(|| coefficients[source]),
                            OrderedBlockLinearStorage::DenseDstSrc(coefficients) => {
                                coefficients[destination * sources.len() + source]
                            }
                        };
                        match block.storage() {
                            OrderedBlockLinearStorage::SingletonColumns { .. } => singleton += 1,
                            OrderedBlockLinearStorage::DenseDstSrc(_) => dense += 1,
                        }
                        for (source, rows) in per_pair.into_iter().enumerate() {
                            let rows = rows.unwrap();
                            for (destination, key) in block.destinations().iter().enumerate() {
                                let expected = rows
                                    .iter()
                                    .filter(|(candidate, _)| candidate == key)
                                    .map(|(_, value)| *value)
                                    .sum::<f64>();
                                let actual = coefficient(destination, source).unwrap_or(0.0);
                                assert!(
                                    (actual - expected).abs() <= 1.0e-12 * (1.0 + expected.abs()),
                                    "{key:?}: block {actual} vs per-pair {expected}"
                                );
                            }
                            for (key, value) in &rows {
                                if value.abs() > 1.0e-12 {
                                    assert!(block.destinations().contains(key), "missing {key:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        singleton > 0 && dense > 0 && braided > 0,
        "singleton {singleton} dense {dense} braided {braided}"
    );
}
