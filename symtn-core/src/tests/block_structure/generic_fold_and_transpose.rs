use super::*;

struct CountingA4FoldRule {
    n_calls: std::cell::Cell<usize>,
    f_calls: std::cell::Cell<usize>,
}

impl FusionRule for CountingA4FoldRule {
    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(&A4FoldRule)
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionRule::fusion_style(&A4FoldRule)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(&A4FoldRule)
    }
    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(&A4FoldRule)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        FusionRule::dual(&A4FoldRule, sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        FusionRule::fusion_channels(&A4FoldRule, left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.n_calls.set(self.n_calls.get() + 1);
        FusionRule::nsymbol(&A4FoldRule, left, right, coupled)
    }
}

impl GenericFusionSymbols for CountingA4FoldRule {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        self.f_calls.set(self.f_calls.get() + 1);
        GenericFusionSymbols::f_symbol_generic(&A4FoldRule, a, b, c, d, e, f)
    }

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericFusionSymbols::r_symbol_generic(&A4FoldRule, a, b, c)
    }
}

// Look up the coeff vector for the multi_Fmove output tree with the given
// coupled sector and (single) vertex label.
fn find_coeff(out: &[(FusionTreeKey, Vec<f64>)], coupled: usize, vtx: usize) -> Vec<f64> {
    out.iter()
        .find(|(tr, _)| tr.coupled().id() == coupled && tr.vertices()[0].get() == vtx)
        .unwrap_or_else(|| panic!("no output tree coupled={coupled} vtx={vtx}"))
        .1
        .clone()
}

fn assert_vec(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length {} != {}",
        got.len(),
        want.len()
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-10, "{label}[{i}]: {g} != {w}");
    }
}

// Gate 4a (A4 oracle): multi_Fmove of every rank-3 A4 (3,3,3)->3 tree matches
// TensorKit.multi_Fmove exactly, INCLUDING the coefficient VECTORS. The
// inner=3 rows exercise the non-trivial F(3,3,3,3,3,3) block, so the vector
// machinery (F-slice selection, μ/ν/κ vertex indexing, λ free axis) is fully
// discriminated here — unlike the A4 bend oracle, whose A/B are I₂.
#[test]
fn a4_multi_fmove_matches_tensorkit() {
    let rule = A4FoldRule;
    let s = 1.0 / 3.0_f64.sqrt();
    let m = -1.0 / (2.0 * 3.0_f64.sqrt());
    let o = 1.0 / 3.0;
    // (inner,v1,v2) -> [(coupled, vtx, coeff)]. TK.multi_Fmove gold values.
    type Row = (usize, usize, usize, Vec<(usize, usize, Vec<f64>)>);
    let table: Vec<Row> = vec![
        (
            0,
            1,
            1,
            vec![
                (0, 1, vec![o]),
                (1, 1, vec![o]),
                (2, 1, vec![o]),
                (3, 1, vec![s, 0.0]),
                (3, 2, vec![0.0, s]),
            ],
        ),
        (
            1,
            1,
            1,
            vec![
                (0, 1, vec![o]),
                (1, 1, vec![o]),
                (2, 1, vec![o]),
                (3, 1, vec![m, -0.5]),
                (3, 2, vec![0.5, m]),
            ],
        ),
        (
            2,
            1,
            1,
            vec![
                (0, 1, vec![o]),
                (1, 1, vec![o]),
                (2, 1, vec![o]),
                (3, 1, vec![m, 0.5]),
                (3, 2, vec![-0.5, m]),
            ],
        ),
        (
            3,
            1,
            1,
            vec![
                (0, 1, vec![s]),
                (1, 1, vec![m]),
                (2, 1, vec![m]),
                (3, 1, vec![0.5, 0.0]),
                (3, 2, vec![0.0, -0.5]),
            ],
        ),
        (
            3,
            1,
            2,
            vec![
                (0, 1, vec![0.0]),
                (1, 1, vec![0.5]),
                (2, 1, vec![-0.5]),
                (3, 1, vec![0.0, -0.5]),
                (3, 2, vec![-0.5, 0.0]),
            ],
        ),
        (
            3,
            2,
            1,
            vec![
                (0, 1, vec![0.0]),
                (1, 1, vec![-0.5]),
                (2, 1, vec![0.5]),
                (3, 1, vec![0.0, -0.5]),
                (3, 2, vec![-0.5, 0.0]),
            ],
        ),
        (
            3,
            2,
            2,
            vec![
                (0, 1, vec![s]),
                (1, 1, vec![m]),
                (2, 1, vec![m]),
                (3, 1, vec![-0.5, 0.0]),
                (3, 2, vec![0.0, 0.5]),
            ],
        ),
    ];
    for (inner, v1, v2, outputs) in table {
        let out = generic_multi_fmove_tree(&rule, &a4f_rank3(inner, v1, v2)).unwrap();
        assert_eq!(out.len(), 5, "in({inner},{v1},{v2}): 5 tails");
        // What: every tail-coupled candidate carries the same externals, and
        // the candidates of one tail-coupled sector share one arena.
        assert!(out.windows(2).all(|terms| {
            terms[0].0.uncoupled() == terms[1].0.uncoupled()
                && terms[0].0.is_dual() == terms[1].0.is_dual()
                && (terms[0].0.coupled() != terms[1].0.coupled()
                    || Arc::ptr_eq(&terms[0].0.arena, &terms[1].0.arena))
        }));
        for (coupled, vtx, want) in outputs {
            let got = find_coeff(&out, coupled, vtx);
            assert_vec(
                &got,
                &want,
                &format!("in({inner},{v1},{v2}) out(c={coupled},v={vtx})"),
            );
        }
    }
}

#[test]
fn infallible_generic_associator_keeps_raw_f_shape_policy() {
    let long = a4f_rank3(3, 2, 1);
    let tail = generic_multi_fmove_tree(&A4FoldRule, &long)
        .unwrap()
        .into_iter()
        .find(|(tree, _)| tree.coupled().id() == 3 && tree.vertices()[0].get() == 1)
        .unwrap()
        .0;
    let probe = CountingA4FoldRule {
        n_calls: std::cell::Cell::new(0),
        f_calls: std::cell::Cell::new(0),
    };
    generic_multi_associator_result(&InfallibleGenericFR(&probe), &long, &tail)
        .unwrap()
        .unwrap();
    assert_eq!(probe.f_calls.get(), 1);
    assert_eq!(
        probe.n_calls.get(),
        1,
        "infallible access must not add four F-shape Nsymbol queries"
    );
}

// A4 provider whose `malformed_call`-th F block keeps its data but reports an
// inconsistent shape, as an infallible provider may.
struct MalformedA4FProbe {
    f_calls: std::cell::Cell<usize>,
    malformed_call: usize,
    malform: fn(GenericFArray<f64>) -> GenericFArray<f64>,
}

impl MalformedA4FProbe {
    fn new(malformed_call: usize, malform: fn(GenericFArray<f64>) -> GenericFArray<f64>) -> Self {
        Self {
            f_calls: std::cell::Cell::new(0),
            malformed_call,
            malform,
        }
    }
}

impl FusionRule for MalformedA4FProbe {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionRule::fusion_style(&A4FoldRule)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(&A4FoldRule)
    }
    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(&A4FoldRule)
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        FusionRule::dual(&A4FoldRule, sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        FusionRule::fusion_channels(&A4FoldRule, left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        FusionRule::nsymbol(&A4FoldRule, left, right, coupled)
    }
}

impl GenericFusionSymbols for MalformedA4FProbe {
    type Scalar = f64;

    fn f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> GenericFArray<Self::Scalar> {
        let call = self.f_calls.get() + 1;
        self.f_calls.set(call);
        let valid = GenericFusionSymbols::f_symbol_generic(&A4FoldRule, a, b, c, d, e, f);
        if call == self.malformed_call {
            (self.malform)(valid)
        } else {
            valid
        }
    }

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericFusionSymbols::r_symbol_generic(&A4FoldRule, a, b, c)
    }
}

impl GenericRigidSymbols for MalformedA4FProbe {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        GenericRigidSymbols::dim_scalar(&A4FoldRule, sector)
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        GenericRigidSymbols::sqrt_dim_scalar(&A4FoldRule, sector)
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        GenericRigidSymbols::inv_sqrt_dim_scalar(&A4FoldRule, sector)
    }
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        GenericRigidSymbols::frobenius_schur_phase_scalar(&A4FoldRule, sector)
    }
}

// (μ, ν, κ, λ) shape reported as (κ, λ, μ, ν) over unchanged row-major data.
fn swap_f_axis_pairs(f: GenericFArray<f64>) -> GenericFArray<f64> {
    let (mu, nu, kappa, lambda) = f.shape();
    GenericFArray::new(f.data().to_vec(), (kappa, lambda, mu, nu))
}

// One extra zero μ row: every in-range read is unchanged.
fn pad_f_mu_axis(f: GenericFArray<f64>) -> GenericFArray<f64> {
    let (mu, nu, kappa, lambda) = f.shape();
    let mut data = f.data().to_vec();
    data.resize(data.len() + nu * kappa * lambda, 0.0);
    GenericFArray::new(data, (mu + 1, nu, kappa, lambda))
}

// Last μ row dropped: the incoming vector is longer than the μ axis.
fn shrink_f_mu_axis(f: GenericFArray<f64>) -> GenericFArray<f64> {
    let (mu, nu, kappa, lambda) = f.shape();
    let mut data = f.data().to_vec();
    data.truncate((mu - 1) * nu * kappa * lambda);
    GenericFArray::new(data, (mu - 1, nu, kappa, lambda))
}

fn assert_malformed_f_shape<T>(
    result: Result<T, CheckedGenericSymbolError<std::convert::Infallible>>,
) {
    assert!(matches!(
        result,
        Err(CheckedGenericSymbolError::Core(
            CoreError::MalformedFusionTree { .. }
        ))
    ));
}

// What: the raw infallible associator rejects a fixed F label outside the
// returned shape instead of reading a neighbouring entry. F(3,3,3,3,1,3) is
// (1,1,2,2); reported as (2,2,1,1), κ = 2 is out of axis while the flat
// index stays in range.
#[test]
fn raw_generic_associator_rejects_f_label_outside_returned_shape() {
    let long = a4f_rank3(1, 1, 1);
    let tail = generic_multi_fmove_tree(&A4FoldRule, &long)
        .unwrap()
        .into_iter()
        .find(|(tree, _)| tree.coupled().id() == 3 && tree.vertices()[0].get() == 2)
        .unwrap()
        .0;
    let valid = MalformedA4FProbe::new(usize::MAX, swap_f_axis_pairs);
    assert!(
        generic_multi_associator_result(&InfallibleGenericFR(&valid), &long, &tail)
            .unwrap()
            .is_some()
    );
    assert_eq!(valid.f_calls.get(), 1);

    let probe = MalformedA4FProbe::new(1, swap_f_axis_pairs);
    assert_malformed_f_shape(generic_multi_associator_result(
        &InfallibleGenericFR(&probe),
        &long,
        &tail,
    ));
}

// What: at k > 2 the incoming coefficient vector must match the μ axis exactly;
// TensorKit's `transpose(view(F, :, ν, κ, :)) * coeff` throws on the same
// length mismatch.
#[test]
fn raw_generic_associator_rejects_coefficient_length_mismatch() {
    let t = SectorId::new(3);
    let label = |n| MultiplicityIndex::new(n).expect("test multiplicity label is one-based");
    let long = FusionTreeKey::new(
        [t, t, t, t],
        t,
        [false; 4],
        [t, t],
        [label(1), label(1), label(1)],
    );
    // Both F slices are F(3,3,3,3,3,3), the modelled rank-4 A4 family.
    let tail = FusionTreeKey::new([t, t, t], t, [false; 3], [t], [label(1), label(1)]);
    let valid = MalformedA4FProbe::new(usize::MAX, pad_f_mu_axis);
    assert!(
        generic_multi_associator_result(&InfallibleGenericFR(&valid), &long, &tail)
            .unwrap()
            .is_some()
    );
    assert_eq!(valid.f_calls.get(), 2);

    for malform in [pad_f_mu_axis, shrink_f_mu_axis] {
        let probe = MalformedA4FProbe::new(2, malform);
        assert_malformed_f_shape(generic_multi_associator_result(
            &InfallibleGenericFR(&probe),
            &long,
            &tail,
        ));
    }
}

// What: the default A/B derivations on the raw infallible path validate the
// whole F shape before their reshape reads.
#[test]
fn raw_generic_a_and_b_reject_malformed_f_shape() {
    let t = SectorId::new(3);
    let bad_b = MalformedA4FProbe::new(1, swap_f_axis_pairs);
    assert!(matches!(
        GenericRigidAccess::try_b_symbol_generic(&InfallibleGenericFR(&bad_b), t, t, t),
        Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
    ));
    let bad_a = MalformedA4FProbe::new(1, swap_f_axis_pairs);
    assert!(matches!(
        GenericRigidAccess::try_a_symbol_generic(&InfallibleGenericFR(&bad_a), t, t, t),
        Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
    ));
}

// Build the full foldright coefficient map keyed by output tree pair. The
// output collapses multiple (codomain', domain') paths per pair (the A-matrix
// contraction) — the accumulator already summed them.
fn foldright_map(
    rule: &A4FoldRule,
    pair: &FusionTreePairKey,
) -> std::collections::HashMap<FusionTreePairKey, f64> {
    let mut map = std::collections::HashMap::new();
    for (out, coeff) in generic_foldright_tree_pair(rule, pair).unwrap() {
        *map.entry(out).or_insert(0.0) += coeff;
    }
    map
}

// Gate 4b (A4 oracle): tree-level foldright U-matrix vs TensorKit.foldright.
// rank-2 codomain: dst domain first leg dualized, U == I₂ (A4 Asymbol=I₂).
#[test]
fn a4_foldright_rank2_matches_tensorkit() {
    let rule = A4FoldRule;
    let t = SectorId::new(3);
    for mu in 1..=2 {
        // src: cod [3,3]->3 (vtx μ), dom [3]->3.
        let cod = FusionTreeKey::new(
            [t, t],
            t,
            [false, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([t], t, [false], [], []);
        let map = foldright_map(&rule, &FusionTreePairKey::pair(cod, dom));
        // TK dst: cod [3]->3, dom [3,3]->3 (isdual=(true,false)) vtx μ, U[μ,μ]=1.
        let exp_cod = FusionTreeKey::new([t], t, [false], [], []);
        let exp_dom = FusionTreeKey::new(
            [t, t],
            t,
            [true, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let exp = FusionTreePairKey::pair(exp_cod, exp_dom);
        for (key, coeff) in &map {
            let want = if key == &exp { 1.0 } else { 0.0 };
            assert!(
                (coeff - want).abs() < 1e-10,
                "rank2 μ={mu}: coeff {coeff} want {want}"
            );
        }
        assert!(
            (map.get(&exp).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
            "rank2 μ={mu} self"
        );
    }
}

// Gate 4b (A4 oracle): rank-3 foldright — the 7×7 U-matrix that fully
// exercises F(3,3,3,3,3,3) combined with the √dim coeff factors (the ±√3/2 =
// ±0.8660 entries). This is the strongest fold discriminator available.
#[test]
fn a4_foldright_rank3_matches_tensorkit() {
    let rule = A4FoldRule;
    let t = SectorId::new(3);
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    // src columns (cod [3,3,3]->3 inner=x vtx=(v1,v2), dom [3]->3).
    let cols: [(usize, usize, usize); 7] = [
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (3, 2, 1),
        (3, 1, 2),
        (3, 2, 2),
    ];
    // dst rows: (cod_coupled, cod_vtx, dom_coupled, dom_vtx). dom isdual=(true,false).
    let rows: [(usize, usize, usize, usize); 7] = [
        (0, 1, 0, 1),
        (1, 1, 1, 1),
        (2, 1, 2, 1),
        (3, 1, 3, 1),
        (3, 2, 3, 1),
        (3, 1, 3, 2),
        (3, 2, 3, 2),
    ];
    let sq = 1.0 / 3.0_f64.sqrt(); // 0.57735
    let hs = 3.0_f64.sqrt() / 2.0; // 0.86603
                                   // TK.foldright U (row,col) nonzeros; zeros elsewhere. 1-based -> 0-based.
    let u: [[f64; 7]; 7] = [
        [sq, sq, sq, 1.0, 0.0, 0.0, 1.0],
        [sq, sq, sq, -0.5, -hs, hs, -0.5],
        [sq, sq, sq, -0.5, hs, -hs, -0.5],
        [sq, -0.5 * sq, -0.5 * sq, 0.5, 0.0, 0.0, -0.5],
        [0.0, 0.5, -0.5, 0.0, -0.5, -0.5, 0.0],
        [0.0, -0.5, 0.5, 0.0, -0.5, -0.5, 0.0],
        [sq, -0.5 * sq, -0.5 * sq, -0.5, 0.0, 0.0, 0.5],
    ];
    for (ci, &(inner, v1, v2)) in cols.iter().enumerate() {
        let cod = a4f_rank3(inner, v1, v2);
        let pair = FusionTreePairKey::pair(cod, dom.clone());
        let map = foldright_map(&rule, &pair);
        for (ri, &(cc, cv, dc, dv)) in rows.iter().enumerate() {
            let ex_cod = FusionTreeKey::new(
                [t, t],
                SectorId::new(cc),
                [false, false],
                [],
                [MultiplicityIndex::new(cv).expect("test multiplicity label is one-based")],
            );
            let ex_dom = FusionTreeKey::new(
                [t, t],
                SectorId::new(dc),
                [true, false],
                [],
                [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
            );
            let key = FusionTreePairKey::pair(ex_cod, ex_dom);
            let got = map.get(&key).copied().unwrap_or(0.0);
            assert!(
                (got - u[ri][ci]).abs() < 1e-10,
                "U[row{ri},col{ci}] (in={inner},{v1},{v2}) got {got} want {}",
                u[ri][ci]
            );
        }
    }
}

// Gate 1: fold round-trip identity. foldright then foldleft returns the
// original pair with coefficient 1 (A-unitarity: A A† = I on the bent
// triple). Enumerated over all rank-3 A4 vertex assignments.
#[test]
fn a4_fold_round_trip_identity() {
    let rule = A4FoldRule;
    let t = SectorId::new(3);
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    for (inner, v1, v2) in [
        (0, 1, 1),
        (1, 1, 1),
        (2, 1, 1),
        (3, 1, 1),
        (3, 2, 1),
        (3, 1, 2),
        (3, 2, 2),
    ] {
        let pair = FusionTreePairKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
        let mut totals = std::collections::HashMap::new();
        for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
            for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        for (key, coeff) in &totals {
            let want = if key == &pair { 1.0 } else { 0.0 };
            assert!(
                (coeff - want).abs() < 1e-10,
                "fold rt in({inner},{v1},{v2}): coeff {coeff} want {want}"
            );
        }
        assert!(
            (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
            "fold rt in({inner},{v1},{v2}): self missing"
        );
    }
}

// Gate 2: cycle round-trip. cycleclockwise then cycleanticlockwise == id.
#[test]
fn a4_cycle_round_trip_identity() {
    let rule = A4FoldRule;
    let t = SectorId::new(3);
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    for (inner, v1, v2) in [(0, 1, 1), (3, 1, 1), (3, 2, 1), (3, 1, 2), (3, 2, 2)] {
        let pair = FusionTreePairKey::pair(a4f_rank3(inner, v1, v2), dom.clone());
        let mut totals = std::collections::HashMap::new();
        for (mid, c1) in generic_cycle_clockwise_tree_pair(&rule, &pair).unwrap() {
            for (out, c2) in generic_cycle_anticlockwise_tree_pair(&rule, &mid).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        for (key, coeff) in &totals {
            let want = if key == &pair { 1.0 } else { 0.0 };
            assert!(
                (coeff - want).abs() < 1e-10,
                "cycle rt in({inner},{v1},{v2}): coeff {coeff} want {want}"
            );
        }
        assert!(
            (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
            "cycle rt in({inner},{v1},{v2}): self missing"
        );
    }
}

// Domain rank ≥ 2. The other generic bend/fold tests use a rank-1 domain; these exercise multi_Fmove_inv on a rank-2 domain (its
// candidates are rank-3, so the associator F-chain runs on the domain side)
// and the rank-2-domain bend surgery. Round-trip identities, all A4 vertex
// assignments enumerated.
#[test]
fn a4_fold_round_trip_domain_rank2() {
    let rule = A4FoldRule;
    let t = SectorId::new(3);
    for cod_mu in 1..=2 {
        for (dom_inner, dv) in [(0, 1), (1, 1), (2, 1), (3, 1), (3, 2)] {
            // cod [3,3]->3 (vtx cod_mu); dom [3,3]->3 inner=dom_inner (vtx dv).
            let cod = FusionTreeKey::new(
                [t, t],
                t,
                [false, false],
                [],
                [MultiplicityIndex::new(cod_mu).expect("test multiplicity label is one-based")],
            );
            let dom = FusionTreeKey::new(
                [t, t],
                t,
                [false, false],
                [],
                [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
            );
            let _ = dom_inner; // rank-2 dom has no innerline; kept for label clarity
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
                for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "fold rt dom-rank2 cod_mu={cod_mu} dv={dv}: {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "fold rt dom-rank2 cod_mu={cod_mu} dv={dv}: self missing"
            );
        }
    }
}

#[test]
fn a4_bend_round_trip_domain_rank2() {
    // Through the checked entries, which also validate every pair.
    let rule = InfallibleGeneric::new(&A4FoldRule);
    let t = SectorId::new(3);
    for cod_mu in 1..=2 {
        for dv in 1..=2 {
            let cod = FusionTreeKey::new(
                [t, t],
                t,
                [false, false],
                [],
                [MultiplicityIndex::new(cod_mu).expect("test multiplicity label is one-based")],
            );
            let dom = FusionTreeKey::new(
                [t, t],
                t,
                [false, false],
                [],
                [MultiplicityIndex::new(dv).expect("test multiplicity label is one-based")],
            );
            let pair = FusionTreePairKey::pair(cod, dom);
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_bendright_tree_pair_checked(&rule, &pair).unwrap() {
                for (out, c2) in generic_bendleft_tree_pair_checked(&rule, &mid).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            for (key, coeff) in &totals {
                let want = if key == &pair { 1.0 } else { 0.0 };
                assert!(
                    (coeff - want).abs() < 1e-10,
                    "bend rt dom-rank2 cod_mu={cod_mu} dv={dv}: {coeff} want {want}"
                );
            }
            assert!(
                (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
                "bend rt dom-rank2 cod_mu={cod_mu} dv={dv}: self missing"
            );
        }
    }
}

// Direct: foldright distributes ROW μ of the COMPLEX A-matrix (=U) to the
// domain vertices ν. coeff(out ν) = coeff0·A[μ,ν] = U[μ,ν]. A missing conj
// or a μ↔ν swap would produce conj(U)/Uᵀ — distinct complex numbers.
#[test]
fn refute_complex_foldright_reads_a_row_unconjugated() {
    let rule = ComplexUnitaryRule;
    let s = SectorId::new(1);
    let u = cx_u();
    // Sanity: U genuinely complex and non-Hermitian.
    assert!(u[1].im.abs() > 0.1, "U must be complex");
    assert!((u[1] - u[2].conj()).norm() > 0.1, "U must be non-Hermitian");
    for mu in 1..=2usize {
        let cod = FusionTreeKey::new(
            [s, s],
            s,
            [false, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([s], s, [false], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        let out = generic_foldright_tree_pair(&rule, &pair).unwrap();
        let mut got = [cx(0.0, 0.0); 2];
        for (key, coeff) in &out {
            let nu = key.domain_tree().vertices()[0].get();
            got[nu - 1] = *coeff;
        }
        for nu in 0..2 {
            let want = u[(mu - 1) * 2 + nu]; // ROW μ of U
            assert!(
                (got[nu] - want).norm() < 1e-10,
                "μ={mu} ν={nu}: {} want ROW-μ {} (conj/transpose?)",
                got[nu],
                want
            );
        }
        // Distinguishable from the conjugated reading.
        let want_conj = u[(mu - 1) * 2].conj();
        assert!(
            (got[0] - want_conj).norm() > 1e-9 || u[(mu - 1) * 2].im.abs() < 1e-12,
            "conj reading coincides — test cannot discriminate"
        );
    }
}

// Round-trip with a COMPLEX unitary A: foldright∘foldleft == id requires
// U U† = I, so the conj in the return fold must be exactly right.
#[test]
fn complex_fold_round_trip_identity() {
    let rule = ComplexUnitaryRule;
    let s = SectorId::new(1);
    for mu in 1..=2usize {
        let cod = FusionTreeKey::new(
            [s, s],
            s,
            [false, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([s], s, [false], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        let mut totals: std::collections::HashMap<FusionTreePairKey, Complex64> =
            std::collections::HashMap::new();
        for (mid, c1) in generic_foldright_tree_pair(&rule, &pair).unwrap() {
            for (out, c2) in generic_foldleft_tree_pair(&rule, &mid).unwrap() {
                *totals.entry(out).or_insert(cx(0.0, 0.0)) += c1 * c2;
            }
        }
        for (key, coeff) in &totals {
            let want = if key == &pair {
                cx(1.0, 0.0)
            } else {
                cx(0.0, 0.0)
            };
            assert!(
                (coeff - want).norm() < 1e-10,
                "cx fold rt μ={mu}: {coeff} want {want}"
            );
        }
        assert!(
            (totals.get(&pair).copied().unwrap_or(cx(0.0, 0.0)) - cx(1.0, 0.0)).norm() < 1e-10,
            "cx fold rt μ={mu}: self missing"
        );
    }
}

// Bend round-trip with a COMPLEX unitary B: bendright∘bendleft == id.
#[test]
fn complex_bend_round_trip_identity() {
    let rule = ComplexUnitaryRule;
    let s = SectorId::new(1);
    for mu in 1..=2usize {
        let cod = FusionTreeKey::new(
            [s, s],
            s,
            [false, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([s], s, [false], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        let mut totals: std::collections::HashMap<FusionTreePairKey, Complex64> =
            std::collections::HashMap::new();
        for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
            for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                *totals.entry(out).or_insert(cx(0.0, 0.0)) += c1 * c2;
            }
        }
        for (key, coeff) in &totals {
            let want = if key == &pair {
                cx(1.0, 0.0)
            } else {
                cx(0.0, 0.0)
            };
            assert!(
                (coeff - want).norm() < 1e-10,
                "cx bend rt μ={mu}: {coeff} want {want}"
            );
        }
    }
}

fn su3_bfwd() -> [[f64; 2]; 2] {
    let e = -1.0 / (2.0 * 2.0_f64.sqrt());
    let g = (7.0_f64 / 8.0).sqrt();
    [[e, -g], [g, e]]
}

// Real-categorical bend oracle: bendright distributes coeff₀·ROW μ of the
// NON-DIAGONAL SU(3) B to the domain vertices ν. A μ↔ν swap would emit
// COLUMN μ; B is non-symmetric (B[0,1]≠B[1,0]) so the two are distinct.
#[test]
fn su3_bendright_uses_b_row_not_column() {
    let rule = Su3BendRule;
    let s42 = SectorId::new(1);
    let s31 = SectorId::new(2);
    let b = su3_bfwd();
    let coeff0 = 15.0_f64.sqrt() / 27.0_f64.sqrt(); // √dim(31)/√dim(42)
    for mu in 1..=2usize {
        // cod [42,31]->31 (vtx μ), dom [31]->31.
        let cod = FusionTreeKey::new(
            [s42, s31],
            s31,
            [false, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([s31], s31, [false], [], []);
        let out = generic_bendright_tree_pair(&rule, &FusionTreePairKey::pair(cod, dom)).unwrap();
        let mut got = [0.0f64; 2];
        for (key, coeff) in &out {
            let nu = key.domain_tree().vertices().last().unwrap().get();
            got[nu - 1] = *coeff;
        }
        for nu in 0..2 {
            let want = coeff0 * b[mu - 1][nu]; // ROW μ
            assert!(
                (got[nu] - want).abs() < 1e-10,
                "μ={mu} ν={nu}: {} want coeff0·ROW-μ {} (transpose ⇒ column)",
                got[nu],
                want
            );
        }
        // Distinguishable from the transposed (column) reading.
        let col = coeff0 * b[if mu == 1 { 1 } else { 0 }][mu - 1];
        assert!((got[0] - col).abs() > 1e-9, "μ={mu}: row/column coincide");
    }
}

// Round-trip with a real non-diagonal SU(3) B: bendright∘bendleft == id
// (B_fwd · B_ret = I₂), exercising the non-trivial off-diagonal mixing.
#[test]
fn su3_bend_round_trip_identity() {
    let rule = Su3BendRule;
    let s42 = SectorId::new(1);
    let s31 = SectorId::new(2);
    for mu in 1..=2usize {
        let cod = FusionTreeKey::new(
            [s42, s31],
            s31,
            [false, false],
            [],
            [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
        );
        let dom = FusionTreeKey::new([s31], s31, [false], [], []);
        let pair = FusionTreePairKey::pair(cod, dom);
        let mut totals = std::collections::HashMap::new();
        for (mid, c1) in generic_bendright_tree_pair(&rule, &pair).unwrap() {
            for (out, c2) in generic_bendleft_tree_pair(&rule, &mid).unwrap() {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        for (key, coeff) in &totals {
            let want = if key == &pair { 1.0 } else { 0.0 };
            assert!(
                (coeff - want).abs() < 1e-10,
                "su3 bend rt μ={mu}: {coeff} want {want}"
            );
        }
        assert!(
            (totals.get(&pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
            "su3 bend rt μ={mu}: self missing"
        );
    }
}

fn assert_identity_term_map(
    got: &HashMap<FusionTreePairKey, f64>,
    self_pair: &FusionTreePairKey,
    label: &str,
) {
    for (key, coeff) in got {
        let want = if key == self_pair { 1.0 } else { 0.0 };
        assert!(
            (coeff - want).abs() < 1e-10,
            "{label}: coeff {coeff} != {want}"
        );
    }
    assert!(
        (got.get(self_pair).copied().unwrap_or(0.0) - 1.0).abs() < 1e-10,
        "{label}: self coefficient missing"
    );
}

// A4 rank-1/rank-1 pair: cod [3]->3, dom [3]->3 (coupled sector 3).
fn a4_pair_rank1_1() -> FusionTreePairKey {
    let t = SectorId::new(3);
    let cod = FusionTreeKey::new([t], t, [false], [], []);
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    FusionTreePairKey::pair(cod, dom)
}

// A4 rank-2/rank-1 pair: cod [3,3]->3 (vtx μ), dom [3]->3 — an
// outer-multiplicity tree pair with N(3,3,3)=2.
fn a4_pair_rank2_1(mu: usize) -> FusionTreePairKey {
    let t = SectorId::new(3);
    let cod = FusionTreeKey::new(
        [t, t],
        t,
        [false, false],
        [],
        [MultiplicityIndex::new(mu).expect("test multiplicity label is one-based")],
    );
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    FusionTreePairKey::pair(cod, dom)
}

/// A pair, a cyclic permutation, its inverse, and a label.
type TransposeRoundTrip = (
    FusionTreePairKey,
    &'static [usize],
    &'static [usize],
    &'static [usize],
    &'static [usize],
    &'static str,
);

// Gate 5: transpose (planar cyclic permutation) round-trips to the
// identity. `generic_transpose_tree_pair_checked` chains repartition + the
// fold/bend A-move (`generic_cycle_*`); a cyclic permutation followed by its
// inverse must return the original pair with coefficient 1. Run on
// A4FoldRule, whose A-move is a genuinely non-diagonal outer-multiplicity
// move, so any coefficient error in the composition breaks the identity.
#[test]
fn generic_transpose_round_trips_to_identity() {
    // The checked entry also validates every pair. A one-step clockwise
    // cycle is undone by the anticlockwise one; the two-step cycle with dual
    // flags is its own inverse.
    let rule = InfallibleGeneric::new(&A4FoldRule);
    let t = SectorId::new(3);
    let rank3_dom1 = |codomain: FusionTreeKey| {
        FusionTreePairKey::pair(codomain, FusionTreeKey::new([t], t, [false], [], []))
    };
    let dual_pair = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [t, t, t],
            t,
            [true, false, true],
            [t],
            [MultiplicityIndex::new(2).unwrap(), MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new([t], t, [true], [], []),
    );
    let cases: [TransposeRoundTrip; 4] = [
        (a4_pair_rank1_1(), &[1], &[0], &[1], &[0], "rank-1 swap"),
        (
            rank3_dom1(a4f_rank3(3, 2, 1)),
            &[1, 2, 3],
            &[0],
            &[3, 0, 1],
            &[2],
            "clockwise",
        ),
        (
            rank3_dom1(a4f_rank3(3, 1, 2)),
            &[3, 0, 1],
            &[2],
            &[1, 2, 3],
            &[0],
            "anticlockwise",
        ),
        (
            dual_pair,
            &[2, 3, 0],
            &[1],
            &[2, 3, 0],
            &[1],
            "two-step dual flags",
        ),
    ];
    for (pair, codomain, domain, inverse_codomain, inverse_domain, label) in cases {
        let forward = generic_transpose_tree_pair_checked(&rule, &pair, codomain, domain).unwrap();
        let mut totals = HashMap::new();
        for (mid, c1) in forward {
            for (out, c2) in
                generic_transpose_tree_pair_checked(&rule, &mid, inverse_codomain, inverse_domain)
                    .unwrap()
            {
                *totals.entry(out).or_insert(0.0) += c1 * c2;
            }
        }
        assert_identity_term_map(&totals, &pair, label);
    }
}

// Gate 6: braid with the IDENTITY permutation is the identity map on an
// outer-multiplicity tree pair. The braid decomposes to zero swaps, so the
// composer runs repartition-to-all-codomain and back (a verified bend
// round-trip) around a no-op braid, plus the tree-pair reconstruction
// closure. Run on A4BendRule (rigid, OM); no braid R-symbol is invoked.
#[test]
fn generic_braid_identity_permutation_is_identity() {
    let rule = A4BendRule;
    for mu in 1..=2 {
        let pair = a4_pair_rank2_1(mu);
        // codomain axes [0,1], domain axis [2]; identity level order.
        let got =
            map_terms(generic_braid_tree_pair(&rule, &pair, &[0, 1], &[2], &[0, 1], &[2]).unwrap());
        assert_identity_term_map(&got, &pair, &format!("A4 braid-id μ={mu}"));
    }
}

// Gate 7: `generic_permute_tree_pair` == `generic_braid_tree_pair` under
// the identity level order (the definitional relation the mult-free path
// relies on), and the symmetric-braiding guard is honored. Uses the identity
// permutation because a non-trivial multi-leg *braid* needs a fully-modeled
// braiding generic rule (the SU(3) provider); the composition itself is a
// line-for-line mirror of the fully-tested
// `multiplicity_free_braid_tree_pair`, and its braid step
// (`generic_braid_tree`) is tested independently by the Artin-braid
// round-trip tests.
#[test]
fn generic_permute_agrees_with_default_level_braid() {
    let rule = A4BendRule; // Bosonic ⇒ symmetric braiding.
    for mu in 1..=2 {
        let pair = a4_pair_rank2_1(mu);
        let permuted = map_terms(generic_permute_tree_pair(&rule, &pair, &[0, 1], &[2]).unwrap());
        // default levels: codomain [0,1], domain [2].
        let braided =
            map_terms(generic_braid_tree_pair(&rule, &pair, &[0, 1], &[2], &[0, 1], &[2]).unwrap());
        assert_term_maps_eq(&permuted, &braided, &format!("A4 permute==braid μ={mu}"));
        // And the identity permutation is a genuine no-op.
        assert_identity_term_map(&permuted, &pair, &format!("A4 permute-id μ={mu}"));
    }
}
