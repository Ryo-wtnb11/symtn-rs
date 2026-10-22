use super::*;

// ===================================================================
// ADVERSARIAL CHECKS: stress the nontrivial-F path that the Artin-braid
// round-trip tests deliberately avoid (F forced = I).
// ===================================================================

// Same structure as UnitaryToyOmRule, but F(a,a,a,a,c,c) — the block the
// index>1 braid actually reads — is a NONTRIVIAL 2x2 rotation (in the
// (mu,kappa) plane), not the identity. A single elementary braid needs no
// hexagon consistency, so we can (a) compare the impl's coefficients to an
// INDEPENDENT re-evaluation of TK's formula written here from scratch, and
// (b) assert the elementary braid matrix is unitary (F,R all unitary).
#[derive(Clone, Copy, Debug)]
struct RefuteOmRule;

impl RefuteOmRule {
    const VACUUM: usize = 0;
    const A: usize = 1;
    const C: usize = 3;
    const R_THETA: f64 = std::f64::consts::PI / 5.0;
    const F_PHI: f64 = std::f64::consts::PI / 7.0; // nontrivial F angle
}

impl symtn_sectors::testing::Provider for RefuteOmRule {}

impl FusionRule for RefuteOmRule {
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
        match (left.id(), right.id()) {
            (Self::VACUUM, x) | (x, Self::VACUUM) => smallvec![SectorId::new(x)],
            (Self::A, Self::A) => smallvec![SectorId::new(Self::C)],
            (Self::A, Self::C) | (Self::C, Self::A) => smallvec![SectorId::new(Self::A)],
            _ => smallvec![SectorId::new(Self::VACUUM)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (Self::A, Self::A, Self::C) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for RefuteOmRule {
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
        let shape = (
            self.nsymbol(a, b, e),
            self.nsymbol(e, c, d),
            self.nsymbol(b, c, f),
            self.nsymbol(a, f, d),
        );
        let ids = (a.id(), b.id(), c.id(), d.id(), e.id(), f.id());
        if ids == (Self::A, Self::A, Self::A, Self::A, Self::C, Self::C) {
            // shape (2,1,2,1); row-major flat idx = mu*2 + kappa. Rotation
            // R_phi in the (mu,kappa) plane: F[mu,0,kappa,0] = Rphi[mu,kappa].
            let (s, co) = Self::F_PHI.sin_cos();
            GenericFArray::new(vec![co, -s, s, co], shape)
        } else {
            // 1x1 blocks = 1.0 on every path these tests touch.
            let n = shape.0 * shape.1 * shape.2 * shape.3;
            GenericFArray::new(vec![1.0; n], shape)
        }
    }
    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        let rows = self.nsymbol(a, b, c);
        let cols = self.nsymbol(b, a, c);
        if (a.id(), b.id(), c.id()) == (Self::A, Self::A, Self::C) {
            let (s, co) = Self::R_THETA.sin_cos();
            GenericRMatrix::new(vec![co, -s, s, co], rows, cols)
        } else {
            GenericRMatrix::new(vec![1.0; rows * cols], rows, cols)
        }
    }
}

fn refute_rank3_tree(vertex1: usize) -> FusionTreeKey {
    let a = SectorId::new(RefuteOmRule::A);
    let c = SectorId::new(RefuteOmRule::C);
    FusionTreeKey::new(
        [a, a, a],
        a,
        [false, false, false],
        [c],
        [
            MultiplicityIndex::new(vertex1).expect("test multiplicity label is one-based"),
            MultiplicityIndex::ONE,
        ],
    )
}

// INDEPENDENT re-evaluation of TK braiding_manipulations.jl:170-194 for the
// rank-3 tree [a,a,a]->a braided at index 1. Written directly from the TK
// source WITHOUT calling generic_artin_braid_at_with_inverse. Returns the
// 2x2 braid matrix M[sigma][mu] over the OM channel (nu=lambda=rho=0 fixed,
// c'=c, n_sigma=n_kappa=2).
// Index-form loops kept on purpose: they mirror the TK formula verbatim.
#[allow(clippy::needless_range_loop)]
fn independent_braid_matrix_index1(rule: &RefuteOmRule, inverse: bool) -> [[f64; 2]; 2] {
    let a = SectorId::new(RefuteOmRule::A);
    let c = SectorId::new(RefuteOmRule::C);
    // TK naming for i>1 (0-based index==1 here): a=inner_ext[i-1]=leg0=a,
    // b=uncoupled[i]=a, c=inner_ext[i]=c, d=uncoupled[i+1]=a, e=coupled=a.
    let (a_s, b_s, c_s, d_s, e_s) = (a, a, c, a, a);
    let c_prime = c; // only channel in a (x) a
    let nu = 0usize; // vertices[i]=1 -> 0-based 0
    let n_sigma = rule.nsymbol(a_s, d_s, c_prime); // 2
    let n_lambda = rule.nsymbol(c_prime, b_s, e_s); // 1
    let n_rho = rule.nsymbol(d_s, c_s, e_s); // 1
    let n_kappa = rule.nsymbol(d_s, a_s, c_prime); // 2
    assert_eq!((n_sigma, n_lambda, n_rho, n_kappa), (2, 1, 1, 2));
    // Rmat1 = inv ? R(d,c,e)' : R(c,d,e); Rmat2 = inv ? R(d,a,c')' : R(a,d,c')
    let rmat1 = if inverse {
        rule.r_symbol_generic(d_s, c_s, e_s)
    } else {
        rule.r_symbol_generic(c_s, d_s, e_s)
    };
    let rmat2 = if inverse {
        rule.r_symbol_generic(d_s, a_s, c_prime)
    } else {
        rule.r_symbol_generic(a_s, d_s, c_prime)
    };
    let fmat = rule.f_symbol_generic(d_s, a_s, b_s, e_s, c_prime, c_s);
    let mut m = [[0.0f64; 2]; 2];
    for mu in 0..2usize {
        for sigma in 0..n_sigma {
            let lambda = 0usize;
            let mut coeff = 0.0f64;
            for rho in 0..n_rho {
                for kappa in 0..n_kappa {
                    // Rmat1[nu,rho] (adjoint => conj(base[rho,nu]))
                    let r1 = if inverse {
                        rmat1.get(rho, nu) // 1x1, conj of real = itself
                    } else {
                        rmat1.get(nu, rho)
                    };
                    // conj(Fmat[kappa,lambda,mu,rho]); real => itself
                    let fc = fmat.get(kappa, lambda, mu, rho);
                    // conj(Rmat2[sigma,kappa]); inv => base[kappa,sigma]
                    let r2 = if inverse {
                        rmat2.get(kappa, sigma)
                    } else {
                        rmat2.get(sigma, kappa)
                    };
                    coeff += r1 * fc * r2;
                }
            }
            m[sigma][mu] = coeff;
        }
    }
    m
}

// Extract the impl's 2x2 braid matrix M[sigma][mu] for the same case.
fn impl_braid_matrix_index1(rule: &RefuteOmRule, inverse: bool) -> [[f64; 2]; 2] {
    let mut m = [[0.0f64; 2]; 2];
    for mu1 in 1..=2usize {
        let tree = refute_rank3_tree(mu1);
        let outs = generic_artin_braid_at_with_inverse(rule, &tree, 1, inverse).unwrap();
        for (out, coeff) in outs {
            assert_eq!(out.innerlines(), &[SectorId::new(RefuteOmRule::C)]);
            assert_eq!(out.vertices()[1].get(), 1, "lambda must be 1");
            let sigma = out.vertices()[0].get() - 1;
            m[sigma][mu1 - 1] = coeff;
        }
    }
    m
}

#[test]
fn refute_impl_matches_independent_tk_formula_nontrivial_f() {
    let rule = RefuteOmRule;
    for &inverse in &[false, true] {
        let indep = independent_braid_matrix_index1(&rule, inverse);
        let got = impl_braid_matrix_index1(&rule, inverse);
        for s in 0..2 {
            for m in 0..2 {
                assert!(
                    (indep[s][m] - got[s][m]).abs() < 1e-12,
                    "inverse={inverse} mismatch at [{s}][{m}]: indep={} impl={}",
                    indep[s][m],
                    got[s][m]
                );
            }
        }
    }
}

#[test]
fn refute_elementary_braid_is_unitary_nontrivial_f() {
    // With all R,F unitary the elementary braid matrix must be unitary,
    // even though F here is a nontrivial rotation (no hexagon needed).
    let rule = RefuteOmRule;
    let m = impl_braid_matrix_index1(&rule, false);
    // M^T M == I
    for i in 0..2 {
        for j in 0..2 {
            let dot: f64 = (0..2).map(|k| m[k][i] * m[k][j]).sum();
            let expect = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expect).abs() < 1e-12,
                "braid matrix not unitary at ({i},{j}): {dot}"
            );
        }
    }
}

#[test]
fn refute_roundtrip_matches_analytic_not_identity_when_f_nontrivial() {
    // For a NON-hexagon-consistent F, forward-then-inverse does NOT recover
    // the input for the index>0 path: the composite = M_inv * M_fwd, which
    // analytically is R(theta)^T . R(phi) . R(theta) . R(phi) = R(2*phi)
    // (commuting SO(2)), = I iff phi=0 (F=I). This is exactly why the braid's
    // own round-trip test forces F=I; it is NOT a bug. Here we confirm the impl
    // reproduces the analytic composite (independent M_inv . M_fwd) and that
    // it deviates from identity by precisely R(2*phi).
    let rule = RefuteOmRule;
    let m_fwd = independent_braid_matrix_index1(&rule, false);
    let m_inv = independent_braid_matrix_index1(&rule, true);
    // Analytic composite M_inv . M_fwd (apply forward, then inverse).
    let mut comp = [[0.0f64; 2]; 2];
    for s in 0..2 {
        for m in 0..2 {
            comp[s][m] = (0..2).map(|t| m_inv[s][t] * m_fwd[t][m]).sum();
        }
    }
    // Impl round-trip matrix over mu -> final sigma.
    let mut impl_rt = [[0.0f64; 2]; 2];
    for mu1 in 1..=2usize {
        let tree = refute_rank3_tree(mu1);
        let mut col: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for (mid, cf) in generic_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap() {
            for (fin, ci) in generic_artin_braid_at_with_inverse(&rule, &mid, 1, true).unwrap() {
                assert_eq!(fin.innerlines(), &[SectorId::new(RefuteOmRule::C)]);
                let sigma = fin.vertices()[0].get() - 1;
                *col.entry(sigma).or_insert(0.0) += cf * ci;
            }
        }
        for (sigma, v) in col {
            impl_rt[sigma][mu1 - 1] = v;
        }
    }
    // (a) impl reproduces the analytic composite.
    for s in 0..2 {
        for m in 0..2 {
            assert!(
                (impl_rt[s][m] - comp[s][m]).abs() < 1e-12,
                "impl round-trip != analytic at [{s}][{m}]: {} vs {}",
                impl_rt[s][m],
                comp[s][m]
            );
        }
    }
    // (b) composite == R(2*phi), NOT identity.
    let (s2, c2) = (2.0 * RefuteOmRule::F_PHI).sin_cos();
    let r2phi = [[c2, -s2], [s2, c2]];
    for s in 0..2 {
        for m in 0..2 {
            assert!((comp[s][m] - r2phi[s][m]).abs() < 1e-12);
        }
    }
    assert!(
        (comp[0][0] - 1.0).abs() > 1e-3,
        "sanity: nontrivial-F round-trip should deviate from identity"
    );
}

// ---- TK numeric oracle: real A4Irrep(3) sector, GenericFusion N=2 ----
//
// Values transcribed from TensorKit.jl v0.17 + TensorKitSectors v0.3.9
// (A4Irrep, FusionStyle == GenericFusion()) — computed by TK's OWN
// artin_braid on a FusionTreeBlock, independent of this port. We model just
// the a=b=c=d=e=c'=3 OM sub-block (the braid coefficient formula is local:
// it reads only F(3,3,3,3,3,3), R(3,3,3), which we transcribe exactly),
// and reproduce it with generic_artin_braid_at_with_inverse on the tree
// [3,3,3,3]->0, inner=[3,3], braided at index 1 (= TK i=2). If the F index
// order [κ,λ,μ,ρ] or any conj/adjoint were transposed, this rich
// (non-diagonal) 4x4 (μ,ν)->(σ,λ) block would not match.
#[derive(Clone, Copy, Debug)]
struct A4SubBlockRule;

impl A4SubBlockRule {
    const VACUUM: usize = 0;
    const THREE: usize = 3;
    // F(3,3,3,3,3,3), row-major over (κ,λ,μ,ρ), dims (2,2,2,2). Verbatim
    // from TK (oracle4.jl FLAT_F_rowmajor_klmr), -0.0 normalised to 0.0.
    const F333333: [f64; 16] = [
        0.5, 0.0, 0.0, -0.5, 0.0, -0.5, -0.5, 0.0, 0.0, -0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.5,
    ];
    // R(3,3,3) = [[-1,0],[0,1]] (TK).
    const R333: [f64; 4] = [-1.0, 0.0, 0.0, 1.0];
}

impl symtn_sectors::testing::Provider for A4SubBlockRule {}

impl FusionRule for A4SubBlockRule {
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
        // Truncated to the OM channel; only fusion_channels(3,3) is read by
        // the index-1 braid (for c' enumeration). 3⊗3 ∋ 3 with N=2 in A4.
        if (left.id(), right.id()) == (Self::THREE, Self::THREE) {
            smallvec![SectorId::new(Self::THREE)]
        } else {
            smallvec![SectorId::new(Self::VACUUM)]
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (Self::THREE, Self::THREE, Self::THREE) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for A4SubBlockRule {
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
        let all3 = [a, b, c, d, e, f].iter().all(|s| s.id() == Self::THREE);
        assert!(all3, "only F(3,3,3,3,3,3) is modelled");
        GenericFArray::new(Self::F333333.to_vec(), (2, 2, 2, 2))
    }
    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        assert_eq!(
            (a.id(), b.id(), c.id()),
            (Self::THREE, Self::THREE, Self::THREE),
            "only R(3,3,3) is modelled"
        );
        GenericRMatrix::new(Self::R333.to_vec(), 2, 2)
    }
}

#[test]
#[allow(clippy::type_complexity)] // oracle table typed to match the TK dump verbatim
fn tk_oracle_a4_generic_braid_matches_tensorkit() {
    let rule = A4SubBlockRule;
    let three = SectorId::new(A4SubBlockRule::THREE);
    let vac = SectorId::new(A4SubBlockRule::VACUUM);
    // TK oracle sub-block (identical for inv=false and inv=true here):
    // (mu,nu) -> (sigma,lambda) => coeff.  [oracle4.jl SUBBLOCK]
    let oracle: [((usize, usize), (usize, usize), f64); 8] = [
        ((1, 1), (1, 1), 0.5),
        ((2, 2), (1, 1), 0.5),
        ((2, 1), (2, 1), 0.5),
        ((1, 2), (2, 1), -0.5),
        ((2, 1), (1, 2), -0.5),
        ((1, 2), (1, 2), 0.5),
        ((1, 1), (2, 2), 0.5),
        ((2, 2), (2, 2), 0.5),
    ];
    for &inverse in &[false, true] {
        // Build impl matrix keyed by ((mu,nu),(sigma,lambda)).
        let mut got: std::collections::HashMap<((usize, usize), (usize, usize)), f64> =
            std::collections::HashMap::new();
        for mu in 1..=2usize {
            for nu in 1..=2usize {
                let tree = FusionTreeKey::new(
                    [three, three, three, three],
                    vac,
                    [false, false, false, false],
                    [three, three],
                    [
                        MultiplicityIndex::new(mu).expect("test multiplicity label is one-based"),
                        MultiplicityIndex::new(nu).expect("test multiplicity label is one-based"),
                        MultiplicityIndex::ONE,
                    ],
                );
                for (out, coeff) in
                    generic_artin_braid_at_with_inverse(&rule, &tree, 1, inverse).unwrap()
                {
                    assert_eq!(out.innerlines(), &[three, three], "c'=3, e=3 unchanged");
                    assert_eq!(out.vertices()[2].get(), 1);
                    let sigma = out.vertices()[0].get();
                    let lambda = out.vertices()[1].get();
                    got.insert(((mu, nu), (sigma, lambda)), coeff);
                }
            }
        }
        // Every oracle entry must be reproduced.
        for &(inp, outp, val) in &oracle {
            let g = got.get(&(inp, outp)).copied().unwrap_or(0.0);
            assert!(
                (g - val).abs() < 1e-10,
                "inv={inverse} {inp:?}->{outp:?}: impl={g} TK={val}"
            );
        }
        // And the impl must produce NO nonzero outside the oracle set.
        for (&(inp, outp), &g) in &got {
            if g.abs() > 1e-10 {
                let known = oracle
                    .iter()
                    .any(|&(i, o, v)| i == inp && o == outp && (v - g).abs() < 1e-10);
                assert!(
                    known,
                    "inv={inverse} spurious nonzero {inp:?}->{outp:?}={g}"
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MisreportedSimpleA4Rule;

impl symtn_sectors::testing::Provider for MisreportedSimpleA4Rule {}

impl FusionRule for MisreportedSimpleA4Rule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(&A4BendRule)
    }

    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(&A4BendRule)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        A4BendRule.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        A4BendRule.fusion_channels(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        A4BendRule.nsymbol(left, right, coupled)
    }
}

impl GenericFusionSymbols for MisreportedSimpleA4Rule {
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
        A4BendRule.f_symbol_generic(a, b, c, d, e, f)
    }

    fn r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        A4BendRule.r_symbol_generic(a, b, c)
    }
}

impl GenericRigidSymbols for MisreportedSimpleA4Rule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        A4BendRule.dim_scalar(sector)
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        A4BendRule.sqrt_dim_scalar(sector)
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        A4BendRule.inv_sqrt_dim_scalar(sector)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        A4BendRule.frobenius_schur_phase_scalar(sector)
    }
}

// cod [3,3,3]->3, inner=[x] (vertices v1,v2), dom [3]->3.
fn a4_pair_rank3(inner: usize, v1: usize, v2: usize) -> FusionTreePairKey {
    let t = a4_three();
    let cod = FusionTreeKey::new(
        [t, t, t],
        t,
        [false, false, false],
        [SectorId::new(inner)],
        [
            MultiplicityIndex::new(v1).expect("test multiplicity label is one-based"),
            MultiplicityIndex::new(v2).expect("test multiplicity label is one-based"),
        ],
    );
    let dom = FusionTreeKey::new([t], t, [false], [], []);
    FusionTreePairKey::pair(cod, dom)
}

#[test]
fn checked_generic_bend_queries_only_the_rigid_data_it_uses() {
    let pair = a4_dual_pair_rank2(1);
    let rule = CheckedA4Spy::new();
    generic_bendright_tree_pair_checked(&rule, &pair).unwrap();
    assert_eq!(rule.dual_calls.get(), 3);
    // Includes the rank-1 domain tree's N(c, 1, c) admission probe,
    // which the multiplicity-free checked validator also makes.
    assert_eq!(rule.n_calls.get(), 8);
    assert_eq!(rule.f_calls.get(), 1);
    assert_eq!(rule.sqrt_calls.get(), 3);
    assert_eq!(rule.inv_sqrt_calls.get(), 2);
    assert_eq!(rule.fs_calls.get(), 1);
}

#[test]
fn checked_generic_bend_stores_the_b_column_as_the_output_vertex() {
    let rule = CheckedA4Spy {
        non_diagonal_b: true,
        ..CheckedA4Spy::new()
    };
    let out = generic_bendright_tree_pair_checked(&rule, &a4_pair_rank2(1)).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(
        out.iter()
            .map(|(key, _)| key.domain_tree().vertices()[0].get())
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
}

#[test]
fn checked_generic_bend_preserves_each_provider_failure_source() {
    let pair = a4_dual_pair_rank2(1);

    let rule = CheckedA4Spy {
        fail_n: Some(2),
        ..CheckedA4Spy::new()
    };
    assert_rigid_provider_error(
        generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
        RigidSpyError::N,
    );

    let rule = CheckedA4Spy {
        fail_dual: Some(2),
        ..CheckedA4Spy::new()
    };
    assert_rigid_provider_error(
        generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
        RigidSpyError::Dual,
    );

    let rule = CheckedA4Spy {
        fail_sqrt: Some(1),
        ..CheckedA4Spy::new()
    };
    assert_rigid_provider_error(
        generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
        RigidSpyError::Sqrt,
    );

    let rule = CheckedA4Spy {
        fail_inv_sqrt: Some(1),
        ..CheckedA4Spy::new()
    };
    assert_rigid_provider_error(
        generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
        RigidSpyError::InvSqrt,
    );

    let rule = CheckedA4Spy {
        fail_fs: Some(1),
        ..CheckedA4Spy::new()
    };
    assert_rigid_provider_error(
        generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
        RigidSpyError::Fs,
    );

    let rule = CheckedA4Spy {
        fail_f: Some(1),
        ..CheckedA4Spy::new()
    };
    assert_rigid_provider_error(
        generic_bendright_tree_pair_checked(&rule, &pair).unwrap_err(),
        RigidSpyError::F,
    );
}

#[test]
fn generic_proof_hook_rechecks_reported_style() {
    // What: the proof-consuming Generic hook rejects a provider that
    // implements Generic symbols but reports a multiplicity-free style.
    let empty = BlockStructure::empty(0);
    let wrong_style =
        LocallyValidatedFusionTreeBlockStructure::try_new(&MisreportedSimpleA4Rule, &empty)
            .unwrap();
    assert_eq!(
        wrong_style
            .generic_permute_tree_pair_for_block_index(0, &[], &[])
            .unwrap_err(),
        CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Simple,
        }
    );
}

fn round_trip_bend(
    rule: &A4BendRule,
    pair: &FusionTreePairKey,
) -> std::collections::HashMap<FusionTreePairKey, f64> {
    let mut totals = std::collections::HashMap::new();
    for (mid, c1) in generic_bendright_tree_pair(rule, pair).unwrap() {
        for (out, c2) in generic_bendleft_tree_pair(rule, &mid).unwrap() {
            *totals.entry(out).or_insert(0.0) += c1 * c2;
        }
    }
    totals
}

fn assert_identity_map(
    totals: &std::collections::HashMap<FusionTreePairKey, f64>,
    expected_self: &FusionTreePairKey,
    label: &str,
) {
    for (key, coeff) in totals {
        let want = if key == expected_self { 1.0 } else { 0.0 };
        assert!(
            (coeff - want).abs() < 1e-12,
            "{label}: coeff {coeff} for is_self={} (want {want})",
            key == expected_self
        );
    }
    assert!(
        (totals.get(expected_self).copied().unwrap_or(0.0) - 1.0).abs() < 1e-12,
        "{label}: self coefficient missing"
    );
}

// Gate 1: bendright∘bendleft == identity (the B-matrix is a Hom-space
// isomorphism), enumerated over all vertex assignments, rank 2 and 3.
#[test]
fn generic_bend_round_trip_identity() {
    let rule = A4BendRule;
    let t = a4_three();
    // Premise the round-trip depends on: N(a,b,c)==N(c,dual(b),a) so the
    // bend is square/invertible on the bent triple (a=b=c=3).
    assert_eq!(rule.nsymbol(t, t, t), rule.nsymbol(t, rule.dual(t), t));
    assert_eq!(rule.nsymbol(t, t, t), 2);

    for mu in 1..=2 {
        let pair = a4_pair_rank2(mu);
        assert_identity_map(
            &round_trip_bend(&rule, &pair),
            &pair,
            &format!("rank2 μ={mu}"),
        );
    }
    // rank 3: inner∈{0,1,2} forces v1=v2=1 (N=1); inner=3 opens both OM
    // vertices v1,v2∈{1,2}. All vertex assignments enumerated.
    for inner in 0..=2 {
        let pair = a4_pair_rank3(inner, 1, 1);
        assert_identity_map(
            &round_trip_bend(&rule, &pair),
            &pair,
            &format!("rank3 inner={inner}"),
        );
    }
    for v1 in 1..=2 {
        for v2 in 1..=2 {
            let pair = a4_pair_rank3(3, v1, v2);
            assert_identity_map(
                &round_trip_bend(&rule, &pair),
                &pair,
                &format!("rank3 inner=3 v=({v1},{v2})"),
            );
        }
    }
}

// Gate 2: repartition(N) then repartition back to the original N == identity.
// via_n=1 exercises one bend each way; via_n=0 exercises two bends each way,
// covering the rank-1-codomain (left_coupled=vacuum) branch of bendright.
#[test]
fn generic_repartition_round_trip_identity() {
    let rule = A4BendRule;
    for via_n in [1usize, 0usize] {
        for mu in 1..=2 {
            let pair = a4_pair_rank2(mu); // codomain rank 2
            let mut totals = std::collections::HashMap::new();
            for (mid, c1) in generic_repartition_tree_pair(&rule, &pair, via_n).unwrap() {
                for (out, c2) in generic_repartition_tree_pair(&rule, &mid, 2).unwrap() {
                    *totals.entry(out).or_insert(0.0) += c1 * c2;
                }
            }
            assert_identity_map(&totals, &pair, &format!("repartition via {via_n} μ={mu}"));
        }
    }
}

// Oracle: the core A/B derivation matches TK's Bsymbol / Asymbol.
#[test]
fn a4_b_and_a_symbol_match_tensorkit() {
    let rule = A4BendRule;
    let t = a4_three();
    // TK: Bsymbol(3,3,3) == I₂, Asymbol(3,3,3) == I₂ (TKS v0.3.6).
    let b = b_symbol(&rule, t, t, t);
    assert_eq!(b.shape(), (2, 2));
    let a = a_symbol(&rule, t, t, t);
    assert_eq!(a.shape(), (2, 2));
    for i in 0..2 {
        for j in 0..2 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (b.get(i, j) - want).abs() < 1e-10,
                "B[{i},{j}]={}",
                b.get(i, j)
            );
            assert!(
                (a.get(i, j) - want).abs() < 1e-10,
                "A[{i},{j}]={}",
                a.get(i, j)
            );
        }
    }
}

// Focused unit test for the domain-empty keep-last overwrite: mirrors TK's
// block assignment `U[row, col] = coeff` (duality_manipulations.jl:110),
// where every ν collapses onto the same output key (no vertex to store) and
// the LAST non-zero ν wins. A4's Bsymbol is diagonal so it never puts two
// non-zeros in one row — this needs a synthetic non-diagonal B, and with an
// empty domain c = 1, so a consistent rule has N(1, b̄, a) <= 1 columns. The
// probe therefore declares N(0,1,1) = 2 and supplies the 2×2×2×1 bend
// F-block whose row 0 is [0.3, 0.7] (all dims 1, so B = F[:, :, 0, 0]).
#[derive(Clone, Copy, Debug)]
struct OverwriteProbeRule;

impl symtn_sectors::testing::Provider for OverwriteProbeRule {}

impl FusionRule for OverwriteProbeRule {
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
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(0)],
            _ => smallvec![SectorId::new(0)],
        }
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if matches!((left.id(), right.id(), coupled.id()), (1, 1, 0) | (0, 1, 1)) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}

impl GenericFusionSymbols for OverwriteProbeRule {
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
        // Row 0 = [0.3, 0.7]: two non-zeros, distinct, so keep-last is
        // distinguishable from keep-first (0.3) and from sum (1.0).
        let mut data = vec![0.0; 8];
        data[0] = 0.3; // F[0,0,0,0]
        data[2] = 0.7; // F[0,1,0,0]
        GenericFArray::new(data, (2, 2, 2, 1))
    }
    fn r_symbol_generic(
        &self,
        _a: SectorId,
        _b: SectorId,
        _c: SectorId,
    ) -> GenericRMatrix<Self::Scalar> {
        GenericRMatrix::new(vec![1.0], 1, 1)
    }
}

impl GenericRigidSymbols for OverwriteProbeRule {
    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[test]
fn generic_bendright_domain_empty_keeps_last_nu() {
    let rule = OverwriteProbeRule;
    let a = SectorId::new(1);
    let c = rule.vacuum();
    // cod [1,1]->vac (vertex 1); dom []->vac (EMPTY domain ⇒ ν has nowhere to go).
    let cod = FusionTreeKey::new([a, a], c, [false, false], [], [MultiplicityIndex::ONE]);
    let dom = FusionTreeKey::new([], c, [], [], []);
    let pair = FusionTreePairKey::pair(cod, dom);
    pair.validate_for_rule(&rule).unwrap();
    let out = generic_bendright_tree_pair(&rule, &pair).unwrap();
    assert_eq!(out.len(), 1, "empty domain collapses ν to one key");
    // coeff0 = √dim(vac)·(1/√dim(1)) = 1; keep-last ⇒ B[0,1] = 0.7.
    assert!(
        (out[0].1 - 0.7).abs() < 1e-12,
        "keep-last ν: got {}",
        out[0].1
    );
}

// Oracle: tree-level bendright tables vs TensorKit's own bendright.
#[test]
fn a4_bendright_tree_table_matches_tensorkit() {
    let rule = A4BendRule;
    let sq3 = 3.0_f64.sqrt();

    // --- rank 2: cod [3,3]->3 (μ), dom [3]->3.  TK (probe5.jl):
    //   μ=1 -> dom vertex 1, coeff 1 ;  μ=2 -> dom vertex 2, coeff 1.
    // The checked entry also validates the pair; repartitioning the rank-2
    // codomain to rank 1 is TensorKit's single bendright.
    for mu in 1..=2 {
        let out = generic_bendright_tree_pair_checked(&rule, &a4_pair_rank2(mu)).unwrap();
        let repartitioned =
            generic_repartition_tree_pair_checked(&rule, &a4_pair_rank2(mu), 1).unwrap();
        assert_eq!(repartitioned, out, "rank2 μ={mu} repartition");
        let nonzero: Vec<_> = out.iter().filter(|(_, c)| c.abs() > 1e-10).collect();
        assert_eq!(nonzero.len(), 1, "rank2 μ={mu} expects one nonzero");
        let (key, coeff) = nonzero[0];
        assert_eq!(key.codomain_tree().uncoupled(), [a4_three()], "rank2 cod");
        assert!(
            key.codomain_tree().vertices().is_empty(),
            "rank2 cod rank-1 no vtx"
        );
        assert_eq!(
            key.domain_tree().vertices()[0].get(),
            mu,
            "rank2 ν == μ (B diagonal)"
        );
        assert!((coeff - 1.0).abs() < 1e-10, "rank2 μ={mu} coeff {coeff}");
    }

    // --- rank 3: cod [3,3,3]->3 inner=[x] (v1,v2), dom [3]->3.
    // TK (probe6.jl): coeff = √dim(3)/√dim(inner) = √3 (inner∈{0,1,2}) or 1
    // (inner=3); output cod vertex = v1, dom vertex = ν = v2 (B=I diagonal).
    // Table rows: (inner, v1, v2) -> (cod_vtx, dom_vtx, coeff).
    let table: [(usize, usize, usize, usize, usize, f64); 7] = [
        (0, 1, 1, 1, 1, sq3),
        (1, 1, 1, 1, 1, sq3),
        (2, 1, 1, 1, 1, sq3),
        (3, 1, 1, 1, 1, 1.0),
        (3, 2, 1, 2, 1, 1.0),
        (3, 1, 2, 1, 2, 1.0),
        (3, 2, 2, 2, 2, 1.0),
    ];
    for (inner, v1, v2, cod_vtx, dom_vtx, coeff) in table {
        let out =
            generic_bendright_tree_pair_checked(&rule, &a4_pair_rank3(inner, v1, v2)).unwrap();
        let nonzero: Vec<_> = out.iter().filter(|(_, c)| c.abs() > 1e-10).collect();
        assert_eq!(nonzero.len(), 1, "rank3 ({inner},{v1},{v2}) one nonzero");
        let (key, got) = nonzero[0];
        let left_coupled = key.codomain_tree().coupled().id();
        assert_eq!(left_coupled, inner, "rank3 left_coupled == innerline");
        assert_eq!(
            key.codomain_tree().vertices()[0].get(),
            cod_vtx,
            "rank3 cod vtx"
        );
        assert_eq!(
            key.domain_tree().vertices()[0].get(),
            dom_vtx,
            "rank3 dom vtx"
        );
        assert!(
            (got - coeff).abs() < 1e-10,
            "rank3 ({inner},{v1},{v2}) coeff {got} want {coeff}"
        );
    }
}

fn tp_expected_a() -> [[f64; 2]; 2] {
    let factor = 2.0 * 2.0 * 0.5;
    let kappa_a = 1.0f64; // FS phase, real
    let mut a = [[0.0; 2]; 2];
    for k in 0..2 {
        for l in 0..2 {
            // conj(κ_a · F[0,0,κ,λ]) · factor; all real here.
            a[k][l] = factor * (kappa_a * TP_FA[k * 2 + l]);
        }
    }
    a
}

#[test]
fn refute_b_symbol_is_not_transposed() {
    let rule = TransposeProbeRule;
    let s = SectorId::new(1);
    let b = b_symbol(&rule, s, s, s);
    assert_eq!(b.shape(), (2, 2));
    let want = tp_expected_b();
    // Sanity: the oracle itself must be non-symmetric, else no discrimination.
    assert!(
        (want[0][1] - want[1][0]).abs() > 0.1,
        "oracle B must be non-symmetric"
    );
    for (mu, row) in want.iter().enumerate() {
        for (nu, &want_value) in row.iter().enumerate() {
            assert!(
                (b.get(mu, nu) - want_value).abs() < 1e-12,
                "B[{mu},{nu}]={} want {} (μ↔ν transpose?)",
                b.get(mu, nu),
                want_value
            );
        }
    }
}

#[test]
fn refute_a_symbol_is_not_transposed() {
    // No other bend test uses the A derivation (fold tests do, through the
    // A4 oracle whose A is I₂), so this is the only check of its κ↔λ index
    // order.
    let rule = TransposeProbeRule;
    let s = SectorId::new(1);
    let a = a_symbol(&rule, s, s, s);
    assert_eq!(a.shape(), (2, 2));
    let want = tp_expected_a();
    assert!(
        (want[0][1] - want[1][0]).abs() > 0.1,
        "oracle A must be non-symmetric"
    );
    for (k, row) in want.iter().enumerate() {
        for (l, &want_value) in row.iter().enumerate() {
            assert!(
                (a.get(k, l) - want_value).abs() < 1e-12,
                "A[{k},{l}]={} want {} (κ↔λ transpose?)",
                a.get(k, l),
                want_value
            );
        }
    }
}

fn b_symbol<R: GenericRigidSymbols>(
    rule: &R,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> GenericRMatrix<R::Scalar> {
    GenericRigidAccess::try_b_symbol_generic(&InfallibleGeneric::new(rule), a, b, c).unwrap()
}

fn a_symbol<R: GenericRigidSymbols>(
    rule: &R,
    a: SectorId,
    b: SectorId,
    c: SectorId,
) -> GenericRMatrix<R::Scalar> {
    GenericRigidAccess::try_a_symbol_generic(&InfallibleGeneric::new(rule), a, b, c).unwrap()
}
